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
# from dotenv import load_dotenv

# # credencias oracle e slack
# load_dotenv('C:/Users/ruan.morais/Desktop/sandbox_freire/forecast_credito_pf/credencials.env') 

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
        		select dt_merge, sum(vlr_total) vlr_total
        		from pdgt_maistodos_credito.fl_report_credito_refactor
        		where id_funil_fluxo >= 7 and financiadoras <> 'dr cash parcelex'
        		and dt_merge <= date('2024-12-31')
        		group by 1
        	),
        	antes_2024_parcelex as (
        		select dt_merge, sum(vlr_requerido) vlr_total
        		from pdgt_maistodos_credito.fl_report_credito_refactor
        		where id_funil_fluxo >= 7 and financiadoras = 'parcelex'
        		and dt_merge <= date('2024-12-31')
        		group by 1
        	),
        	depois_2025_sem_parcelex as (
        		select dt_merge, sum(vlr_total) vlr_total
        		from pdgt_maistodos_credito.fl_report_credito_refactor
        		where id_funil_fluxo = 7 and financiadoras <> 'dr cash parcelex'
        		and dt_merge >= date('2025-01-01')
        		group by 1
        	),
        	depois_2025_parcelex as (
        		select dt_merge, sum(vlr_requerido) vlr_total
        		from pdgt_maistodos_credito.fl_report_credito_refactor
        		where id_funil_fluxo = 7 and financiadoras = 'parcelex'
        		and dt_merge >= date('2025-01-01')
        		group by 1
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
        	sum(vlr_total) AS vlr_total
        from uniao
        group by 1
        order by 1
        """

        credit = aws_fx_instance.fx_exec_query(query)
        credit['dt_merge'] = pd.to_datetime(credit['dt_merge'], format='%Y-%m-%d')
        credit['vlr_total'] = credit['vlr_total'].astype(float)

        credit['year'] = credit['dt_merge'].dt.year
        credit['month'] = credit['dt_merge'].dt.month
        credit['day'] = credit['dt_merge'].dt.dayofyear
        credit['quarter/year'] = credit['dt_merge'].dt.to_period('Q').astype(str)
        credit['year_seasonality'] = credit['vlr_total'] / credit.groupby(['year'])['vlr_total'].transform('sum')
        credit.sort_values(by='dt_merge', ascending=True, inplace=True)

        # ========================
        # Preparação p/ Prophet
        # ========================
        df = credit[['dt_merge', 'vlr_total']].rename(columns={'dt_merge': 'ds', 'vlr_total': 'y'})

        model = Prophet(daily_seasonality=True, interval_width=0.95)
        model.add_country_holidays(country_name='BR')
        model.add_seasonality(name='weekly', period=7, fourier_order=5)
        model.fit(df)

        today = datetime.now(timezone.utc)
        end_date = datetime(2025, 12, 31, tzinfo=timezone.utc)
        periods = max((end_date - today).days, 0)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        forecast['anual_seasonality'] = forecast['yhat'] / forecast.groupby(pd.Grouper(key='ds', freq='YE'))['yhat'].transform('sum')
        forecast['quarter_seasonality'] = forecast['yhat'] / forecast.groupby(pd.Grouper(key='ds', freq='QE'))['yhat'].transform('sum')
        forecast['meta_anual'] = np.where(
            forecast['ds'].dt.year == 2025, forecast['anual_seasonality'] * 145_000_000,
            np.where(forecast['ds'].dt.year == 2024, forecast['anual_seasonality'] * 105_000_000, np.nan)
        )

        # ========================
        # Merge com histórico
        # ========================
        forecast.rename(columns={'ds': 'date'}, inplace=True)
        credit = credit.rename(columns={'dt_merge': 'date'})
        forecast = pd.merge(credit, forecast, on='date', how='right').rename(columns={'y': 'vlr_total'})

        # ========================
        # Upload para o S3
        # ========================
        csv_buffer = io.StringIO()
        forecast.to_csv(csv_buffer, index=False)

        s3_client = get_s3_client()
        s3_client.put_object(
            Bucket='todos-data-lake-external-source',
            Key='source=csv/database=business-analytics/forecast_credito_pf/forecast_credito.csv',
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )

        print('Upload realizado com sucesso no S3!')

    except Exception as e:
        print("Erro durante execução do script:")
        import traceback
        traceback.print_exc()
        exit(1)
