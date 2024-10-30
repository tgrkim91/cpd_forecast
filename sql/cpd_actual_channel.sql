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
    tg.signup_month,
    case when a.platform in ('Android native','Desktop web','iOS native','Mobile web') then a.platform
        else 'Undefined' end as platform,
        -- This groupings for segment A are rescaled risk bins correct?
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
        , NVL(f.protection_total ,0)
            + NVL(f.liability_total ,0)
            + NVL(f.customer_support ,0)
            + NVL(f.payment_processing ,0)
            + NVL(f.incidental_bad_debt ,0)
            + NVL(f.chargeback_plus_fee ,0)   AS total_costs
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
    -- drop reservations where trip dates were before their signup date (errors in data)
    and tg.signup_date <= rs.trip_end_ts
;

drop table if exists #cost_per_trip_day_raw;
select channels,
    signup_month,
    increments_from_signup,
    count(reservation_id) as reservations,
    sum(paid_days) as paid_days,
    sum(total_costs)/sum(paid_days) as total_cost_per_day
into #cost_per_trip_day_raw
from #trips_all
group by 1,2,3
order by 2,1,3;

select *
from #cost_per_trip_day_raw;