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

-------- all trips --------------------------------
DROP TABLE IF EXISTS #trips_all_demand;
SELECT
        s.signup_month,
        rs.driver_id,
        s.channels,
        rd.monaco,
        case when b.platform in ('Android native','Desktop web','iOS native','Mobile web') then b.platform
            else 'Undefined' end as platform,
        date_trunc('month', rd.created) as created_month,
        date_trunc('month', rs.trip_end_ts) as trip_end_month,
        case when rd.created::TIMESTAMP < '2023-04-19 18:28:00' and rd.monaco < 0.01 then 'A1' 
            when rd.created::TIMESTAMP >= '2023-04-19 18:28:00' and rd.monaco < 0.0033 then 'A1'
            when rd.created::TIMESTAMP < '2023-04-19 18:28:00' and rd.monaco < 0.02 then 'A2'
            when rd.created::TIMESTAMP >= '2023-04-19 18:28:00' and rd.monaco < 0.0075 then 'A2'
            when rd.created::TIMESTAMP < '2023-04-19 18:28:00' and rd.monaco < 0.03 then 'A3'
            when rd.created::TIMESTAMP >= '2023-04-19 18:28:00' and rd.monaco < 0.0135 then 'A3'
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
        ELSE 'NA' END AS monaco_bin,
        rs.reservation_id,
        rd.country as trip_country,
        DATE_TRUNC('day', rs.trip_end_ts)::D AS trip_end_date,
        rd.paid_days
INTO #trips_all_demand
FROM analytics.reservation_summary rs
INNER JOIN #temp_signup_base s 
    ON rs.driver_id = s.driver_id
INNER JOIN analytics.reservation_dimensions rd
    ON rd.reservation_id = rs.reservation_id
left join (select driver_id,max(platform) as platform from marketing.marketing_channel_by_driver group by driver_id) b
    on rs.driver_id=b.driver_id
WHERE rs.current_status NOT IN (2,11) AND rs.is_ever_booked=1
    AND date_trunc('month',rs.trip_start_ts)>='2017-01-01'
    AND date_trunc('month',rs.trip_end_ts) <=dateadd('month',-1,date_trunc('month',current_date));

with tmp as (
    SELECT channels
        , trip_end_month
        , monaco_bin
        , sum(paid_days) as paid_days
    FROM #trips_all_demand
    group by 1, 2, 3
)

select a.channels 
    , a.trip_end_month
    , a.monaco_bin
    , a.paid_days
    , a.paid_days/b.paid_days_total as distribution_full
from tmp as a
left join (
    select channels
        , trip_end_month
        , sum(paid_days) as paid_days_total
    FROM #trips_all_demand
    group by 1, 2
) as b
on a.trip_end_month = b.trip_end_month and a.channels=b.channels
order by 2, 1, 3;