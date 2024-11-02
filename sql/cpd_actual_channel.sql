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
    and nvl(sucf.rank_desc_paid_90_days, sucf.rank_desc_90_days) = 1
;

drop table if exists #data_warehouse_signups_pick;
select *,
       1 as pick_rank
into #data_warehouse_signups_pick
from #data_warehouse_signups
where 1=1
    and country = 'US'
    and channels in ('Apple_Brand', 'Facebook/IG_App', 'Facebook/IG_Web', 'Facebook_Free')
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

select *
into #trips_all
from 
    (select rs.driver_id,
        tg.channels,
        case when tg.channels in ('Kayak_Desktop', 'Kayak_Desktop_Carousel', 'Kayak_Afterclick', 'Kayak_Desktop_Compare', 'Kayak_Desktop_Front_Door') then 'Kayak_Desktop_Ad'
            when tg.channels in ('Kayak_Mobile_Carousel', 'Kayak_Mobile', 'Kayak_Mobile_Front_Door') then 'Kayak_Mobile_Ad'
            when tg.channels in ('Mediaalpha','Expedia', 'Kayak_Desktop', 'Kayak_Desktop_Carousel', 'Kayak_Afterclick', 'Kayak_Desktop_Compare', 
                'Kayak_Desktop_Front_Door', 'Kayak_Mobile_Carousel', 'Kayak_Mobile', 'Kayak_Mobile_Front_Door',
                'Kayak_Desktop_Core', 'Kayak_Mobile_Core') then 'all_travel_agency'
            when tg.channels in ('Facebook/IG_Web', 'Facebook/IG_App', 'Facebook_Free', 'Reddit') then 'all_social_media'
            else tg.channels end as segment,
        tg.signup_month,
        case when a.platform in ('Android native','Desktop web','iOS native','Mobile web') then a.platform
            else 'Undefined' end as platform_t,
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
        FLOOR(DATEDIFF(day, tg.signup_date, trip_end_date)/30) + 1 as increments_from_signup, 
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
                + NVL(f.chargeback_plus_fee ,0)   AS total_costs
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
        -- drop reservations where trip dates were before their signup date (errors in data)
        and tg.signup_date <= rs.trip_end_ts)
union all
    (select rs.driver_id,
        tg.channels,
        'all_app' as segment,
        tg.signup_month,
        case when a.platform in ('Android native','Desktop web','iOS native','Mobile web') then a.platform
            else 'Undefined' end as platform_t,
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
        FLOOR(DATEDIFF(day, tg.signup_date, trip_end_date)/30) + 1 as increments_from_signup, 
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
                + NVL(f.chargeback_plus_fee ,0)   AS total_costs
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
        -- drop reservations where trip dates were before their signup date (errors in data)
        and tg.signup_date <= rs.trip_end_ts
        and platform_t in ('Android native','iOS native'))
union all
    (select rs.driver_id,
        tg.channels,
        tg.channels as segment,
        tg.signup_month,
        case when a.platform in ('Android native','Desktop web','iOS native','Mobile web') then a.platform
            else 'Undefined' end as platform_t,
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
        FLOOR(DATEDIFF(day, tg.signup_date, trip_end_date)/30) + 1 as increments_from_signup, 
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
                + NVL(f.chargeback_plus_fee ,0)   AS total_costs
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
        -- drop reservations where trip dates were before their signup date (errors in data)
        and tg.signup_date <= rs.trip_end_ts
        and tg.channels in ('Kayak_Desktop_Core', 'Kayak_Mobile_Core'))
;

drop table if exists #cost_per_trip_day_raw;
select segment,
    signup_month,
    increments_from_signup,
    count(reservation_id) as reservations,
    sum(paid_days) as paid_days,
    sum(total_costs)/sum(paid_days) as total_cost_per_day
into #cost_per_trip_day_raw
from #trips_all
group by 1,2,3
order by 2,1,3;

SELECT a.channels,
    b.*                                                                                          
FROM (SELECT channels,
        -- mapping table from channel to segment
        case when channels in ('Kayak_Desktop', 'Kayak_Desktop_Carousel', 'Kayak_Afterclick', 'Kayak_Desktop_Compare', 'Kayak_Desktop_Front_Door') then 'Kayak_Desktop_Ad'
            when channels in ('Kayak_Mobile_Carousel', 'Kayak_Mobile', 'Kayak_Mobile_Front_Door') then 'Kayak_Mobile_Ad'
            when channels in ('Mediaalpha', 'Expedia') then 'all_travel_agency'
            when channels in ('Apple') then 'all_app'
            when channels in ('Facebook/IG_Web', 'Facebook/IG_App', 'Reddit') then 'all_social_media'
            else channels end as segment
    FROM 
    (SELECT distinct channels from #trips_all)) as a
LEFT JOIN #cost_per_trip_day_raw as b on a.segment = b.segment
where a.channels in ('Apple', 'Apple_Brand', 'Google_Desktop','Google_Desktop_Brand',
            'Google_Mobile','Google_Mobile_Brand','Kayak_Desktop', 'Kayak_Desktop_Core', 
            'Kayak_Mobile_Core','Mediaalpha','Expedia','Microsoft_Desktop',
            'Microsoft_Desktop_Brand', 'Reddit', 'Moloco', 'Kayak_Desktop_Compare',
            'Google_Pmax','Kayak_Desktop_Carousel','Kayak_Mobile_Carousel',
            'Kayak_Afterclick', 'Facebook/IG_App', 'Facebook/IG_Web');