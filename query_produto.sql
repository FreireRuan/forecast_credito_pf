with
        	antes_2024_sem_parcelex as (
        		select dt_merge, produto, sum(vlr_total) vlr_total
        		from pdgt_maistodos_credito.fl_report_credito_refactor
        		where id_funil_fluxo >= 7 and financiadoras <> 'dr cash parcelex'
        		and dt_merge <= date('2024-12-31')
        		group by 1, 2
        	),
        	antes_2024_parcelex as (
        		select dt_merge, produto, sum(vlr_requerido) vlr_total
        		from pdgt_maistodos_credito.fl_report_credito_refactor
        		where id_funil_fluxo >= 7 and financiadoras = 'parcelex'
        		and dt_merge <= date('2024-12-31')
        		group by 1, 2
        	),
        	depois_2025_sem_parcelex as (
        		select dt_merge, produto, sum(vlr_total) vlr_total
        		from pdgt_maistodos_credito.fl_report_credito_refactor
        		where id_funil_fluxo = 7 and financiadoras <> 'dr cash parcelex'
        		and dt_merge >= date('2025-01-01')
        		group by 1, 2
        	),
        	depois_2025_parcelex as (
        		select dt_merge, produto, sum(vlr_requerido) vlr_total
        		from pdgt_maistodos_credito.fl_report_credito_refactor
        		where id_funil_fluxo = 7 and financiadoras = 'parcelex'
        		and dt_merge >= date('2025-01-01')
        		group by 1, 2
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
            produto,
        	sum(vlr_total) AS vlr_total
        from uniao
        group by 1, 2
        order by 1