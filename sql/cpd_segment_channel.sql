-- US signups
-- all channels
-- all trips (inside US + outside US)
-- distribution by trip_end_month
drop table if exists #attribution_v3_signups;
select driver_id,
       signup_date,
       signup_month,
       case when country='AU' and channel_lvl_5 in ('Google_Desktop','Google_Mobile') then 'Google'
            when country='AU' and channel_lvl_5 in ('Google_Desktop_Brand','Google_Mobile_Brand') then 'Google_Brand'
            else channel_lvl_5 end as channels,
       platform,
       country,
       2 as pick_rank
into #attribution_v3_signups
from marketing.marketing_channel_by_driver
where country in ('US')
and channel_lvl_5<>'Unknown';


drop table if exists #data_warehouse_signups;
select ddi.driver_id,
       dd.date as signup_date,
       date_trunc('month', dd.date)::d as signup_month,
       cd.channel_name as channels,
       sf.platform,
       ld.country_code as country
into #data_warehouse_signups
from warehouse.session_fact sf
join warehouse.date_dim dd
    on sf.session_date_pdt = dd.date
join warehouse.location_dim ld
    on sf.source_location_key = ld.location_key
join warehouse.channel_dim cd
    on sf.channel_key = cd.channel_key
join warehouse.sign_up_conversion_mart sucf
    on sf.session_key = sucf.session_key
join warehouse.driver_dim ddi
    on sucf.driver_key=ddi.driver_key
where true
    and sucf.rank_desc_paid_90_days = 1
;

drop table if exists #data_warehouse_signups_pick;
select *,
       1 as pick_rank
into #data_warehouse_signups_pick
from #data_warehouse_signups
where 1=1
    and country = 'US'
    and channels = 'Apple_Brand'
;

drop table if exists #temp_signup_base;
select *
into #temp_signup_base
from
    (select *,
           row_number() over (partition by driver_id order by pick_rank) as rank1
    from
        (select *
            from #attribution_v3_signups
        union all
            (select * from #data_warehouse_signups_pick)))
where rank1=1;


--#trips_all
drop table if exists #trips_all;

select
        rs.driver_id,
        tg.channels,
        case when a.platform in ('Android native','Desktop web','iOS native','Mobile web') then a.platform
            else 'Undefined' end as platform,
        CASE WHEN rd.created::TIMESTAMP < '2023-04-19 18:28:00' AND rd.monaco < 0.01 THEN 'A1'
             WHEN rd.created::TIMESTAMP >= '2023-04-19 18:28:00' AND rd.monaco <= 0.0033 THEN 'A1'
             WHEN rd.created::TIMESTAMP < '2023-04-19 18:28:00' AND rd.monaco < 0.02 THEN 'A2'
             WHEN rd.created::TIMESTAMP >= '2023-04-19 18:28:00' AND rd.monaco <= 0.0075 THEN 'A2'
             WHEN rd.created::TIMESTAMP < '2023-04-19 18:28:00' AND rd.monaco < 0.03 THEN 'A3'
             WHEN rd.created::TIMESTAMP >= '2023-04-19 18:28:00' AND rd.monaco <= 0.0135 THEN 'A3'
             WHEN rd.created::TIMESTAMP < '2023-04-19 18:28:00' AND rd.monaco < 0.06 THEN 'B'
             WHEN rd.created::TIMESTAMP >= '2023-04-19 18:28:00' AND rd.monaco <= 0.0315 THEN 'B'
             WHEN rd.created::TIMESTAMP < '2023-04-19 18:28:00' AND rd.monaco < 0.09 THEN 'C'
             WHEN rd.created::TIMESTAMP >= '2023-04-19 18:28:00' AND rd.monaco <= 0.05 THEN 'C'
             WHEN rd.created::TIMESTAMP < '2023-04-19 18:28:00' AND rd.monaco < 0.12 THEN 'D'
             WHEN rd.created::TIMESTAMP >= '2023-04-19 18:28:00' AND rd.monaco <= 0.07 THEN 'D'
             WHEN rd.created::TIMESTAMP < '2023-04-19 18:28:00' AND rd.monaco < 0.18 THEN 'E'
             WHEN rd.created::TIMESTAMP >= '2023-04-19 18:28:00' AND rd.monaco <= 0.10 THEN 'E'
             WHEN rd.created::TIMESTAMP < '2023-04-19 18:28:00' AND rd.monaco >= 0.18 THEN 'F'
             WHEN rd.created::TIMESTAMP >= '2023-04-19 18:28:00' AND rd.monaco > 0.10 THEN 'F'
             WHEN rd.monaco IS NULL THEN 'NA'
            ELSE 'NA' END as monaco_bin,
        rs.reservation_id,
        date_trunc('day', rs.trip_end_ts)::D as trip_end_date,
        rd.paid_days,
        case when c.reservation_id is not null then 1 else 0 end as is_claim
         ,COALESCE(f.gaap_net_revenue,0) as net_revenue
         ,COALESCE(f.partial_contribution_profit - f.gaap_net_revenue * 0.02,0) as contribution_profit
         ,COALESCE(rs.gross_revenue_usd,0) as gross_revenue
         , NVL(f.protection_total, 0) AS protection
         , NVL(f.liability_total, 0)  AS liability
         , NVL(f.customer_support, 0) AS customer_support_cost
         , NVL(f.payment_processing, 0) AS payment_processing_hosting
         , NVL(f.incidental_bad_debt, 0) AS incidental_bad_debt
         , NVL(f.chargeback_plus_fee, 0) AS chargebacks
         , NVL(f.valet, 0) AS valet
         , NVL(f.protection_total ,0)
             + NVL(f.liability_total ,0)
             + NVL(f.customer_support ,0)
             + NVL(f.payment_processing ,0)
             + NVL(f.incidental_bad_debt ,0)
             + NVL(f.chargeback_plus_fee ,0)
             + NVL(f.valet ,0)   AS total_costs
into #trips_all
from analytics.reservation_summary rs
inner join #temp_signup_base tg
    on rs.driver_id=tg.driver_id
inner join finance.reservation_profit_summary_staging f
    on f.reservation_id = rs.reservation_id
inner join analytics.reservation_dimensions rd
    on rd.reservation_id = rs.reservation_id
left join (select driver_id,max(platform) as platform from marketing.marketing_channel_by_driver group by driver_id) a
    on rs.driver_id=a.driver_id
left join (select reservation_id from analytics.claim_summary_rebuilt where paid_claims=1 group by reservation_id) c
    on rs.reservation_id=c.reservation_id
where rs.current_status not in (2,11) and rs.is_ever_booked=1
    and date_trunc('month',rs.trip_start_ts)>='2017-01-01'
    and date_trunc('month',rs.trip_end_ts) <=dateadd('month',-1,date_trunc('month',current_date))
    and (tg.channels<>'Tiktok' or (tg.channels='Tiktok' and tg.signup_month>='2022-07-01'))
;

--cost per trip day_raw
drop table if exists #cost_per_trip_day_raw;
select channels,
       platform,
       monaco_bin,
       date_trunc('month', trip_end_date)::D as trip_end_month,
       count(reservation_id) as reservations,
       sum(paid_days) as paid_days,
       sum(gross_revenue) as gross_revenue,
       sum(net_revenue) as net_revenue,
       sum(contribution_profit) as contribution_profit,
       sum(total_costs) as total_costs,
       sum(protection) as protection,
       sum(liability) as liability,
       sum(customer_support_cost) as customer_support_cost,
       sum(payment_processing_hosting) as payment_processing_hosting,
       sum(incidental_bad_debt) as incidental_bad_debt,
       sum(chargebacks) as chargebacks,
       sum(valet) as valet,
       sum(is_claim) as num_claims
into #cost_per_trip_day_raw
from #trips_all
group by 1,2,3,4
order by 1,2,3,4;

--cost per trip day_pivot
drop table if exists #cost_per_trip_day_by_monaco_temp1;
select *
into #cost_per_trip_day_by_monaco_temp1
from
    (select analytics_month,
            case when channels in ('Free_Google','Free_Microsoft','Free_Other') then 'all free' else channels end as segment,
            monaco_bin,
           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then protection else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as protection_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then liability else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as liability_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then customer_support_cost else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as customer_support_cost_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then payment_processing_hosting else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as payment_processing_hosting_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then incidental_bad_debt else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as incidental_bad_debt_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then chargebacks else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as chargebacks_per_day
    from #cost_per_trip_day_raw a
    join (select date_trunc('month',date) as analytics_month
         from analytics.date
         where date_trunc('day',date)>='2018-01-01' and date_trunc('day',date)<=date_trunc('day',current_date)
         group by 1) b
    on 1=1
    where a.channels is not null
    group by 1,2,3)
union all
   (select analytics_month,
            'all paid' as segment,
            monaco_bin,
           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then protection else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as protection_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then liability else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as liability_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then customer_support_cost else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as customer_support_cost_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then payment_processing_hosting else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as payment_processing_hosting_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then incidental_bad_debt else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as incidental_bad_debt_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then chargebacks else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as chargebacks_per_day
    from #cost_per_trip_day_raw a
    join (select date_trunc('month',date) as analytics_month
         from analytics.date
         where date_trunc('day',date)>='2018-01-01' and date_trunc('day',date)<=date_trunc('day',current_date)
         group by 1) b
    on 1=1
    where a.channels not in ('Free_Google','Free_Microsoft','Free_Other')
    group by 1,2,3)
union all
   (select analytics_month,
            case when platform='Desktop web' then 'all desktop'
             when platform='Mobile web' then 'all mobile'
             when platform='Android native' then 'all android'
             when platform='iOS native' then 'all ios' end as segment,
            monaco_bin,
           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then protection else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as protection_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then liability else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as liability_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then customer_support_cost else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as customer_support_cost_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then payment_processing_hosting else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as payment_processing_hosting_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then incidental_bad_debt else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as incidental_bad_debt_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then chargebacks else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as chargebacks_per_day
    from #cost_per_trip_day_raw a
    join (select date_trunc('month',date) as analytics_month
         from analytics.date
         where date_trunc('day',date)>='2018-01-01' and date_trunc('day',date)<=date_trunc('day',current_date)
         group by 1) b
    on 1=1
    group by 1,2,3)
union all
   (select analytics_month,
            'all web' as segment,
            monaco_bin,
           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then protection else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as protection_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then liability else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as liability_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then customer_support_cost else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as customer_support_cost_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then payment_processing_hosting else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as payment_processing_hosting_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then incidental_bad_debt else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as incidental_bad_debt_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then chargebacks else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as chargebacks_per_day
    from #cost_per_trip_day_raw a
    join (select date_trunc('month',date) as analytics_month
         from analytics.date
         where date_trunc('day',date)>='2018-01-01' and date_trunc('day',date)<=date_trunc('day',current_date)
         group by 1) b
    on 1=1
    where a.platform in ('Desktop web','Mobile web')
    group by 1,2,3)
union all
   (select analytics_month,
            'all app' as segment,
            monaco_bin,
           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then protection else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as protection_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then liability else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as liability_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then customer_support_cost else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as customer_support_cost_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then payment_processing_hosting else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as payment_processing_hosting_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then incidental_bad_debt else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as incidental_bad_debt_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then chargebacks else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as chargebacks_per_day
    from #cost_per_trip_day_raw a
    join (select date_trunc('month',date) as analytics_month
         from analytics.date
         where date_trunc('day',date)>='2018-01-01' and date_trunc('day',date)<=date_trunc('day',current_date)
         group by 1) b
    on 1=1
    where a.platform in ('Android native','iOS native')
    group by 1,2,3)
union all
   (select analytics_month,
            'all google' as segment,
            monaco_bin,
           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then protection else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as protection_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then liability else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as liability_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then customer_support_cost else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as customer_support_cost_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then payment_processing_hosting else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as payment_processing_hosting_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then incidental_bad_debt else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as incidental_bad_debt_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then chargebacks else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as chargebacks_per_day
    from #cost_per_trip_day_raw a
    join (select date_trunc('month',date) as analytics_month
         from analytics.date
         where date_trunc('day',date)>='2018-01-01' and date_trunc('day',date)<=date_trunc('day',current_date)
         group by 1) b
    on 1=1
    where a.channels in ('Google_Desktop','Google_Mobile')
    group by 1,2,3)
union all
   (select analytics_month,
            'all' as segment,
            monaco_bin,
           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then protection else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as protection_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then liability else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as liability_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then customer_support_cost else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as customer_support_cost_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then payment_processing_hosting else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as payment_processing_hosting_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then incidental_bad_debt else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as incidental_bad_debt_per_day,

           sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then chargebacks else 0 end)
           / nullif(sum(case when trip_end_month <= dateadd('month',-4,analytics_month)
                and  trip_end_month >= dateadd('month',-11,analytics_month)
           then paid_days else 0 end),0) as chargebacks_per_day
    from #cost_per_trip_day_raw a
    join (select date_trunc('month',date) as analytics_month
         from analytics.date
         where date_trunc('day',date)>='2018-01-01' and date_trunc('day',date)<=date_trunc('day',current_date)
         group by 1) b
    on 1=1
    group by 1,2,3)
order by 1;

drop table if exists #cost_per_trip_day_by_monaco_temp2;
select *,
       protection_per_day::float + liability_per_day::float + customer_support_cost_per_day::float + payment_processing_hosting_per_day::float
       + incidental_bad_debt_per_day::float + chargebacks_per_day::float as total_cost_per_trip_day
into #cost_per_trip_day_by_monaco_temp2
from #cost_per_trip_day_by_monaco_temp1;

drop table if exists #cost_per_trip_day_by_monaco;
select b.*,
       a.channels
into #cost_per_trip_day_by_monaco
from
(select channels,
       case when channels='Google_Desktop' then 'Google_Desktop'
            when channels='Google_Mobile' then 'Google_Mobile'
            when channels in ('Google_UAC_Android','Google_UAC_iOS') then 'all app'
            when channels='Kayak_Desktop_Core' then 'all desktop'
            when channels in('Free_Google','Free_Microsoft','Free_Other') then 'all free'
            when channels in ('Google_Discovery','Microsoft_Desktop','Kayak_Desktop_Front_Door','Kayak_Desktop_Compare',
                              'Kayak_Desktop','Google_Desktop_Brand','Autorental_Desktop') then 'all desktop'
            when channels in ('Microsoft_Mobile','Google_Mobile_Brand','Kayak_Mobile_Core','Autorental_Mobile','Kayak_Mobile','Hopper',
                              'Kayak_Mobile_Front_Door') then 'all mobile'
            when channels in ('Kayak_Carousel','Delta','Capital_One','Mediaalpha','Expedia') then 'all web'
            when channels in ('Tiktok','Apple','Snapchat') then 'all app'
            else 'all paid' end as segment
from
(select distinct channels from #temp_signup_base)) a
left join #cost_per_trip_day_by_monaco_temp2 b
on a.segment=b.segment;

select *
from #cost_per_trip_day_by_monaco;