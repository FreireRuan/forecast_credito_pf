# ================================
# Bibliotecas
# ================================
import os
import io
import boto3
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone
from prophet import Prophet
from pyathena import connect
from pyathena.pandas.util import as_pandas

# ================================
# Variáveis de ambiente (GitHub Secrets)
# ================================
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
ATHENA_OUTPUT = os.getenv("ATHENA_OUTPUT")
AWS_ACCESS_KEY_ID_BUSINESS_ANALYTICS = os.getenv("AWS_ACCESS_KEY_ID_BUSINESS_ANALYTICS")
AWS_SECRET_ACCESS_KEY_BUSINESS_ANALYTICS = os.getenv("AWS_SECRET_ACCESS_KEY_BUSINESS_ANALYTICS")

# ================================
# Conexões AWS
# ================================
def get_athena_client():
    return boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID_BUSINESS_ANALYTICS,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY_BUSINESS_ANALYTICS,
        region_name=AWS_REGION
    )

# ================================
# Classe utilitária Athena
# ================================
class AWS_fx:
    def __init__(self):
        self.s3_staging_dir = ATHENA_OUTPUT

    def fx_connect_cursor(self):
        return connect(
            s3_staging_dir=self.s3_staging_dir,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        ).cursor()

    def fx_exec_query(self, query):
        cursor = self.fx_connect_cursor()
        cursor.execute(query)
        return as_pandas(cursor)

# ================================
# Execução principal
# ================================
if __name__ == "__main__":
    try:
        print("sessão iniciada")

        aws_fx_instance = AWS_fx()

        # ========================
        # Query de base histórica
        # ========================
        query = """
        with
        	antes_2024_sem_parcelex as (
        		select 
                    dt_merge, 
                    financiadoras,
                    sum(vlr_total) vlr_total
        		from 
                    pdgt_maistodos_credito.fl_report_credito_refactor
        		where 
                    id_funil_fluxo >= 7 
                    and 
                        financiadoras not in ('dr cash parcelex', 'upp', 'b2e legado', 'losango', 'nupay', 'openco', 'dr cash')
        		    and 
                        dt_merge <= date('2024-12-31')
        		group by 
                    1, 2
        	),
        	antes_2024_parcelex as (
        		select 
                    dt_merge,
                    financiadoras,
                    sum(vlr_requerido) vlr_total
        		from 
                    pdgt_maistodos_credito.fl_report_credito_refactor
        		where 
                    id_funil_fluxo >= 7 
                    and 
                        financiadoras = 'parcelex'
        		    and 
                        dt_merge <= date('2024-12-31')
        		group by 
                    1, 2
        	),
        	depois_2025_sem_parcelex as (
        		select 
                    dt_merge, 
                    financiadoras,
                    sum(vlr_total) vlr_total
        		from 
                    pdgt_maistodos_credito.fl_report_credito_refactor
        		where 
                    id_funil_fluxo = 7 
                    and 
                        financiadoras not in ('dr cash parcelex', 'dr cash parcelex', 'upp', 'b2e legado', 'losango', 'nupay', 'openco', 'dr cash')
        		    and 
                        dt_merge >= date('2025-01-01')
        		group by 
                    1, 2
        	),
        	depois_2025_parcelex as (
        		select 
                    dt_merge, 
                    financiadoras,
                    sum(vlr_requerido) vlr_total
        		from 
                    pdgt_maistodos_credito.fl_report_credito_refactor
        		where 
                    id_funil_fluxo = 7 
                    and 
                        financiadoras = 'parcelex'
        		    and 
                        dt_merge >= date('2025-01-01')
        		group by 
                    1, 2
        	),
        	uniao as (
        		select * from antes_2024_sem_parcelex
        		union all
        		select * from antes_2024_parcelex
        		union all
        		select * from depois_2025_sem_parcelex
        		union all
        		select * from depois_2025_parcelex
        	)
        select
        	dt_merge,
            financiadoras,
        	sum(vlr_total) as vlr_total
        from uniao
        group by 
            1, 2
        order by 
            1
        """

        credit = aws_fx_instance.fx_exec_query(query)
        credit['dt_merge'] = pd.to_datetime(credit['dt_merge'], format='%Y-%m-%d')
        credit['vlr_total'] = credit['vlr_total'].astype(float)

        # ========================
        # Preparação de colunas
        # ========================
        credit['year'] = credit['dt_merge'].dt.year
        credit['month'] = credit['dt_merge'].dt.month
        credit['day_of_year'] = credit['dt_merge'].dt.dayofyear
        credit['quarter_year'] = credit['dt_merge'].dt.to_period('Q').astype(str)
        credit.sort_values(by=['dt_merge','financiadoras'], inplace=True)

        # ========================
        # Preparação p/ Prophet
        # ========================
        today = datetime.now(timezone.utc)
        end_date = datetime(2025, 12, 31, tzinfo=timezone.utc)
        periods = max((end_date - today).days, 0)

        forecasts = []
        for lender in credit['financiadoras'].unique():
            df_hist = (
                credit.loc[credit['financiadoras'] == lender, ['dt_merge','vlr_total']]
                      .rename(columns={'dt_merge':'ds','vlr_total':'y'})
            )
            m = Prophet(daily_seasonality=True, interval_width=0.95)
            m.add_country_holidays(country_name='BR')
            m.add_seasonality(name='weekly', period=7, fourier_order=5)
            m.fit(df_hist)

            future = m.make_future_dataframe(periods=periods)
            fc = m.predict(future)[['ds','yhat','yhat_lower','yhat_upper']]

            # Sazonalidades
            fc['anual_seasonality'] = (
                fc['yhat'] / fc.groupby(pd.Grouper(key='ds', freq='YE'))['yhat'].transform('sum')
            )
            fc['quarter_seasonality'] = (
                fc['yhat'] / fc.groupby(pd.Grouper(key='ds', freq='QE'))['yhat'].transform('sum')
            )
            # Metas anuais
            fc['meta_anual'] = np.where(
                fc['ds'].dt.year == 2025, fc['anual_seasonality'] * 145_000_000,
                np.where(fc['ds'].dt.year == 2024, fc['anual_seasonality'] * 105_000_000, np.nan)
            )
            fc['financiadoras'] = lender
            forecasts.append(fc.rename(columns={'ds':'date'}))

        # Concatena forecasts
        forecast_all = pd.concat(forecasts, ignore_index=True)

        # Merge com histórico
        result = pd.merge(
            credit.rename(columns={'dt_merge':'date'}),
            forecast_all,
            on=['date','financiadoras'],
            how='right'
        )

        # ========================
        # Upload para o S3
        # ========================
        csv_buffer = io.StringIO()
        result.to_csv(csv_buffer, index=False)

        s3_client = get_s3_client()
        s3_client.put_object(
            Bucket='todos-data-lake-external-source',
            Key='source=csv/database=business-analytics/forecast_product_credito_pf/forecast_product_credito.csv',
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )

        print('Upload realizado com sucesso no S3!')

    except Exception as e:
        print("Erro durante execução do script:")
        import traceback
        traceback.print_exc()
        exit(1)