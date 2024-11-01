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
where country in ('US','GB','CA','AU')
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

DROP TABLE IF EXISTS #temp_guest_activation;
SELECT DISTINCT a.driver_id
     , FIRST_VALUE(reservation_id)  OVER (PARTITION BY a.driver_id ORDER BY created       ROWS UNBOUNDED PRECEDING)       AS first_booking_reservation_id
     , FIRST_VALUE(reservation_id)  OVER (PARTITION BY a.driver_id ORDER BY trip_start_ts ROWS UNBOUNDED PRECEDING)       AS activation_reservation_id
     , FIRST_VALUE(created)         OVER (PARTITION BY a.driver_id ORDER BY created       ROWS UNBOUNDED PRECEDING)::D    AS first_booking_date
     , FIRST_VALUE(trip_start_ts)   OVER (PARTITION BY a.driver_id ORDER BY trip_start_ts ROWS UNBOUNDED PRECEDING)::D    AS activation_date
INTO #temp_guest_activation
FROM analytics.reservation_summary a
inner join #temp_signup_base b
on a.driver_id=b.driver_id
WHERE 1 = 1
  AND is_ever_booked IN (1)
  AND current_status NOT IN (2, 11)
  and trip_start_ts IS NOT NULL
;

DROP TABLE IF EXISTS #temp_guest_base;
SELECT *
INTO #temp_guest_base
FROM (
    select sb.driver_id
        , sb.signup_date
        , sb.signup_month
        , sb.channels
        , case when sb.channels in ('Kayak_Desktop', 'Kayak_Desktop_Carousel', 'Kayak_Afterclick', 'Kayak_Desktop_Compare', 'Kayak_Desktop_Front_Door') then 'Kayak_Desktop_Ad'
            when sb.channels in ('Kayak_Mobile_Carousel', 'Kayak_Mobile', 'Kayak_Mobile_Front_Door') then 'Kayak_Mobile_Ad'
            when sb.channels in ('Facebook/IG_App','Apple') then 'all_app'
            when sb.channels in ('Facebook/IG_Web', 'Mediaalpha','Expedia', 'Reddit') then 'all web'
            else sb.channels end as segment
        , sb.platform
        , sb.country
        , ga.first_booking_date::D                        AS first_booking_date
        , DATE_TRUNC('month', ga.first_booking_date)::D   AS first_booking_month
        , ga.activation_date::D                           AS activation_date
        , DATE_TRUNC('month', ga.activation_date)::D      AS activation_month
    from #temp_signup_base sb
    LEFT JOIN #temp_guest_activation ga
        ON sb.driver_id = ga.driver_id
    -- filter to drivers from whom we observe two full increments
    where DATEADD('day', 60, sb.signup_date)::Date < CURRENT_DATE - 1
);

DROP TABLE IF EXISTS #temp_guest_group_map;
SELECT *
     , DENSE_RANK() OVER (ORDER BY
       country
     , segment
     , signup_month
     ) AS group_id
INTO #temp_guest_group_map
FROM #temp_guest_base
;

DROP TABLE IF EXISTS #temp_guest_group;
SELECT a.group_id
     , a.country
     , a.segment
     , a.signup_month
     , b.signups
INTO #temp_guest_group
FROM #temp_guest_group_map a
left join
    (select country,segment,signup_month,count(distinct driver_id) as signups
        from #temp_guest_group_map
        group by 1,2,3) b
on a.country=b.country and a.segment=b.segment and a.signup_month=b.signup_month
GROUP BY 1, 2, 3, 4, 5
;

DROP TABLE IF EXISTS #temp_guest_group_add_activations;
SELECT group_id
     , country
     , segment
     , signup_month
     , FLOOR(DATEDIFF(day, signup_date, activation_date)/30) + 1 AS increments_from_signup
     , count(distinct driver_id) as activations
INTO #temp_guest_group_add_activations
FROM #temp_guest_group_map
where activation_date is not null
group by 1,2,3,4,5;

DROP TABLE IF EXISTS #temp_guest_revenue_summary;
select t1.*,
       t2.activations,
       sum(t2.activations) over (partition by t2.group_id order by t2.increments_from_signup ROWS UNBOUNDED PRECEDING) as accumulated_activations
INTO #temp_guest_revenue_summary
from
    (SELECT gm.group_id
         , FLOOR(DATEDIFF(day, gm.signup_date, grb.date::date)/30) + 1 AS increments_from_signup
         , SUM(grb.gross_revenue_usd)  AS gross_revenue_usd
         , SUM(CASE WHEN grb.revenue_category IN ('rental_revenue', 'rental_revenue_discount', 'rental_revenue_boost') THEN grb.gross_revenue_usd ELSE 0 END)                       AS trip_revenue_usd
         , SUM(grb.host_payment_usd)                                                                                                                                                AS host_earnings_usd
         , SUM(grb.net_revenue_usd)                                                                                                                                                 AS net_revenue_usd
         , SUM(CASE WHEN rs.is_ever_booked = 1 AND rs.current_status NOT IN (2, 11) AND grb.revenue_category = 'rental_revenue' THEN booked_noc_days ELSE 0 END)                    AS booked_noc_days
         , SUM(CASE WHEN rs.is_ever_booked = 1 AND rs.current_status NOT IN (2, 11) AND grb.revenue_category = 'rental_revenue' THEN paid_days ELSE 0 END)                          AS paid_days
         , SUM(CASE WHEN rs.is_ever_booked = 1 AND rs.current_status NOT IN (2, 11) AND grb.revenue_category = 'rental_revenue' THEN gaap_trip_count ELSE 0 END)                    AS trips
         , COUNT(DISTINCT CASE WHEN rs.is_ever_booked = 1 AND rs.current_status NOT IN (2, 11) AND grb.revenue_category = 'rental_revenue' THEN grb.reservation_id || grb.date END) AS trip_days
         , COUNT(DISTINCT CASE WHEN rs.is_ever_booked = 1 AND rs.current_status NOT IN (2, 11) AND grb.revenue_category = 'rental_revenue' THEN rs.driver_id END)                   AS active_guests

    FROM finance.gaap_revenue_breakdown grb
             JOIN analytics.reservation_summary rs
                  ON rs.reservation_id = grb.reservation_id
             JOIN #temp_guest_group_map gm
                  ON rs.driver_id = gm.driver_id
    WHERE 1 = 1
      and grb.date>=gm.activation_date
      AND grb.date < CURRENT_DATE - 1
      AND DATEADD('day', 60, gm.signup_date)::Date < CURRENT_DATE - 1
    GROUP BY 1, 2) t1
left join #temp_guest_group_add_activations t2
on t1.group_id=t2.group_id and t1.increments_from_signup=t2.increments_from_signup
;


DROP TABLE IF EXISTS #temp_guest_cost_summary;
SELECT gm.group_id
     , FLOOR(DATEDIFF(day, gm.signup_date, d.date::date)/30) + 1 AS increments_from_signup
     , SUM(NVL(ps.protection_total      , 0) / rd.paid_days::FLOAT)     AS protection_cost
     , SUM(NVL(ps.liability_total       , 0) / rd.paid_days::FLOAT)     AS cost_liability
     , SUM(NVL(ps.customer_support      , 0) / rd.paid_days::FLOAT)     AS customer_support_cost
     , SUM(NVL(ps.payment_processing    , 0) / rd.paid_days::FLOAT)     AS payment_processing_cost
     , SUM(NVL(ps.incidental_bad_debt   , 0) / rd.paid_days::FLOAT)     AS incidental_bad_debt
     , SUM(NVL(ps.chargeback_plus_fee   , 0) / rd.paid_days::FLOAT)     AS chargebacks
     , SUM(NVL(ps.valet                 , 0) / rd.paid_days::FLOAT)     AS valet
     , SUM((NVL(ps.protection_total     , 0)
         + NVL(ps.liability_total       , 0)
         + NVL(ps.customer_support      , 0)
         + NVL(ps.payment_processing    , 0)
         + NVL(ps.incidental_bad_debt   , 0)
         + NVL(ps.chargeback_plus_fee   , 0)
         + NVL(ps.valet                 , 0)) / rd.paid_days::FLOAT)    AS total_cost
INTO #temp_guest_cost_summary
FROM finance.reservation_profit_summary ps -- how many months
         JOIN analytics.reservation_summary rs
              ON ps.reservation_id = rs.reservation_id
         JOIN analytics.reservation_dimensions rd
              ON rs.reservation_id = rd.reservation_id
         JOIN #temp_guest_group_map gm
              ON rs.driver_id = gm.driver_id
         JOIN analytics.date d
              ON d.date BETWEEN COALESCE(CASE WHEN rs.current_status IN (2, 11) THEN rs.modified::D END, rs.current_start_ts::D)
                            AND COALESCE(CASE WHEN rs.current_status IN (2, 11) THEN rs.modified::D END, DATEADD('day', ABS(rd.paid_days)::INT - 1, rs.current_start_ts::D))
WHERE 1 = 1
  and d.date>=gm.activation_date
  AND COALESCE(CASE WHEN rs.current_status IN (2, 11) THEN rs.modified::DATE END, d.date) < CURRENT_DATE - 1
  AND DATEADD('day', 60, gm.signup_date)::Date < CURRENT_DATE - 1
GROUP BY 1, 2
;

drop table if exists #temp_search_session;
select gm.group_id
     , FLOOR(DATEDIFF(day, gm.signup_date, sd.created::date)/30) + 1 AS increments_from_signup
     , count(sd.session_id) as search_sessions
into #temp_search_session
from analytics.session_dimensions sd
         JOIN #temp_guest_group_map gm
              ON sd.driver_id = gm.driver_id
WHERE 1 = 1
  and sd.search=1
  and sd.created>=gm.signup_date
  AND sd.created < CURRENT_DATE - 1
  AND DATEADD('day', 60, gm.signup_date)::Date < CURRENT_DATE - 1
group by 1,2
;

DROP TABLE IF EXISTS #guest_cohort_summary;
SELECT DENSE_RANK() OVER (ORDER BY gg.country, gg.segment, gg.signup_month) AS cohort_id
     , gg.group_id
     , gg.country
     , gg.segment
     , gg.signup_month
     , gg.signups
     , grs.increments_from_signup
     , NVL(gross_revenue_usd, 0)                        AS gross_revenue_usd
     , NVL(trip_revenue_usd, 0)                         AS trip_revenue_usd
     , NVL(host_earnings_usd, 0)                        AS host_earnings_usd
     , NVL(net_revenue_usd, 0)                          AS net_revenue_usd
     , NVL(booked_noc_days, 0)                          AS booked_noc_days
     , NVL(paid_days, 0)                                AS paid_days
     , NVL(trips, 0)                                    AS trips
     , NVL(trip_days, 0)                                AS trip_days
     , NVL(active_guests, 0)                            AS active_guests
     , NVL(activations,0)                               AS activations
     , NVL(accumulated_activations,0)                   AS accumulated_activations
     , NVL(total_cost, 0)                               AS total_cost
     , NVL(net_revenue_usd, 0) - NVL(total_cost, 0)     AS adjusted_contribution
     , NVL(ss.search_sessions,0)                        AS search_sessions
INTO #guest_cohort_summary
FROM #temp_guest_group gg
         LEFT JOIN #temp_guest_revenue_summary grs
                   ON gg.group_id = grs.group_id
         LEFT JOIN #temp_guest_cost_summary gcs
                   ON grs.group_id = gcs.group_id
                       and grs.increments_from_signup=gcs.increments_from_signup
         left join #temp_search_session ss
                   on grs.group_id=ss.group_id
                       and grs.increments_from_signup=ss.increments_from_signup
;

------------------------------------------------------build #marketing_cost--------------------------------------------------------------
drop table if exists #marketing_cost;
select country,
       signup_month,
       channel_lvl_5 as channels,
       sum(calculated_marketing_cost) as marketing_cost
into #marketing_cost
from marketing.marketing_cps_cpa_by_channels
group by 1,2,3
;


drop table if exists #historical_paid_days_per_signup;
select t1.*
into #historical_paid_days_per_signup
from
(select cohort_id,
       country,
       segment,
       signup_month,
       increments_from_signup,
       signups,
       sum(paid_days) as paid_days,
       sum(net_revenue_usd) as net_revenue,
       sum(adjusted_contribution) as adjusted_contribution,
       sum(total_cost) as cost,
       sum(trips) as trips,
       sum(activations) as activations,
       sum(accumulated_activations) as accumulated_activations,
       sum(search_sessions) as search_sessions
from #guest_cohort_summary a
where 1=1
--and increments_from_signup<=12
group by 1,2,3,4,5,6
order by 1,2,3,4,5,6) t1
;


---------------------------------trip days from data science model---------------------------------------
--prediction for US only
drop table if exists #temp_signup_base_us_only;
select *
into #temp_signup_base_us_only
from (
    select *,
        case when channels in ('Kayak_Desktop', 'Kayak_Desktop_Carousel', 'Kayak_Afterclick', 'Kayak_Desktop_Compare', 'Kayak_Desktop_Front_Door') then 'Kayak_Desktop_Ad'
            when channels in ('Kayak_Mobile_Carousel', 'Kayak_Mobile', 'Kayak_Mobile_Front_Door') then 'Kayak_Mobile_Ad'
            when channels in ('Facebook/IG_App','Apple') then 'all_app'
            when channels in ('Facebook/IG_Web', 'Mediaalpha','Expedia', 'Reddit') then 'all web'
            else channels end as segment
    from #temp_signup_base
    where country='US'
);

--ltr
drop table if exists #ltr;
select a.driver_id
     , b.signup_date
     , b.signup_month
     , b.channels
     , b.segment
     , a.days_since_first_trip_end/28 as month_since_first_trip
     , a.ltr
     ,date_trunc('day',a.prediction_date)::D as prediction_date
into #ltr
from
(select *
     from marketing.ltr_from_activation_last
     where run_id in (SELECT run_id FROM marketing_scratch.ltr_run_id)
      ) a

inner join
(select date_trunc('month',prediction_date) as prediction_month,
        max(date_trunc('day',prediction_date)) as prediction_date
 from marketing.ltr_from_activation_last
where run_id in (SELECT run_id FROM marketing_scratch.ltr_run_id)
 group by 1) c
on date_trunc('day',a.prediction_date)=c.prediction_date

inner join #temp_signup_base_us_only b
on a.driver_id=b.driver_id

inner join marketing.ltr_from_activation_historical lha
on lha.driver_id=a.driver_id AND lha.run_id=a.run_id
--averaging by signup day, with 30 day offset
where DATEDIFF(day, b.signup_date, date_trunc('day',a.prediction_date))<=60 and DATEDIFF(day, b.signup_date, date_trunc('day',a.prediction_date))>=30;

drop table if exists #ltr1;
select segment,
       date_trunc('month',dateadd('month',2,prediction_date)) as estimation_month,
       month_since_first_trip,
       avg(ltr) as ltr_per_activation
into #ltr1
from #ltr
group by 1,2,3;

--activation
drop table if exists #activation;
select
         a.driver_id
       , b.signup_date
       , b.signup_month
       , b.channels
       , b.segment
       , a.pred_day_after_signup/28 as month_since_signup
       , a.pred_p as activate_prob
       , date_trunc('day',a.created)::D as prediction_date
into #activation
from
(select driver_id,
        pred_day_after_signup,
        pred_p,
        case when date_trunc('day',created)='2021-10-01' then '2021-09-30' 
             when date_trunc('day',created)='2023-09-01' then '2023-08-31' else created end as created
 from marketing.prediction_driver_activation_curve
 where pred_day_after_signup%28=0 and pred_day_after_signup<>0
    and pred_p>=0 and pred_p is not null
    and model_version IN (SELECT model_version FROM marketing_scratch.activation_run_id)
 ) a

 inner join
 (select date_trunc('month',created) as prediction_month,
         max(date_trunc('day',created)) as prediction_date
  from (select case when date_trunc('day',created)='2021-10-01' then '2021-09-30'
                    when date_trunc('day',created)='2023-09-01' then '2023-08-31' else created end as created
        from marketing.prediction_driver_activation_curve
        where model_version IN (SELECT model_version FROM marketing_scratch.activation_run_id))
  group by 1) c
 on date_trunc('day',a.created)=c.prediction_date

inner join #temp_signup_base_us_only b
on a.driver_id=b.driver_id
-- For each prediction, only include predictions for records between 30-60 days post-signup at time of prediction
where DATEDIFF(day, b.signup_date, date_trunc('day',a.created))<=60 and DATEDIFF(day, b.signup_date, date_trunc('day',a.created))>=30;--averaging by signup date, with 30 day offset


drop table if exists #activation1;
select segment,
       estimation_month,
       month_since_signup,
       num_signups_activation,
       cumulative_activate_rate,
       (case when month_since_signup>1 then cumulative_activate_rate-previous_activate_rate else cumulative_activate_rate end) as incremental_activate_rate
into #activation1
from
    (select *,
           lag(cumulative_activate_rate,1) over (partition by segment,estimation_month order by month_since_signup) as previous_activate_rate
    from
        (select segment,
               date_trunc('month',dateadd('month',2,prediction_date)) as estimation_month,
               month_since_signup,
               avg(activate_prob) as cumulative_activate_rate,
               count(distinct driver_id) as num_signups_activation
        from #activation
        group by 1,2,3));


--ds_master
drop table if exists #activation3;
select a.segment,
       a.signup_month,
       a.num_signups,
       b.month_since_signup,
       dateadd('month',b.month_since_signup-1,a.signup_month) as activation_month,
       b.incremental_activate_rate,
       a.num_signups*b.cumulative_activate_rate as cumulative_activations,
       a.num_signups*b.incremental_activate_rate as activations
into #activation3
from (select *
      from (select segment,
                   signup_month,
                   count(1) as num_signups
            from #temp_signup_base_us_only
            group by 1, 2)
            union all
            (select distinct segment,
                    dateadd('month',1,date_trunc('month',CURRENT_DATE-2)) as signup_month,
                    1000 as num_signups
            from #temp_signup_base_us_only)
     ) a
left join #activation1 b
on a.segment=b.segment
and a.signup_month=b.estimation_month;


drop table if exists #ds_master;
select a.segment,
       a.signup_month,
       a.num_signups,
       a.month_since_signup,
       a.activation_month,
       a.cumulative_activations,
       a.activations,
       b.month_since_first_trip as month_since_activation,
       b.ltr_per_activation,
       dateadd('month',b.month_since_first_trip-1,a.activation_month) as ltr_month,
       datediff('month',a.signup_month,dateadd('month',b.month_since_first_trip-1,a.activation_month)) as ltr_month_since_singup,
       a.activations*b.ltr_per_activation as ltr,
       a.activations*b.ltr_per_activation/a.num_signups as ltr_per_signup
into #ds_master
from #activation3 a
left join #ltr1 b
on a.segment=b.segment
and a.signup_month=b.estimation_month;


drop table if exists #ds_final;
select a.*,
       b.activations,
       a.accumulated_ltr_per_signup-nvl(lag(a.accumulated_ltr_per_signup,1) over (partition by a.segment,a.signup_month order by a.increments_from_signup),0) as ltr_per_signup
into #ds_final
from
    (select segment,
           signup_month,
           ltr_month_since_singup+1 as increments_from_signup,
           ltr_month as trip_month,
           sum(ltr_per_signup) as accumulated_ltr_per_signup
    from #ds_master
    where ltr_month_since_singup+1<=24
    group by 1,2,3,4) a
join
    (select segment,
           signup_month,
           month_since_signup as increments_from_signup,
           avg(activations) as activations
    from #ds_master
    where ltr_month_since_singup+1<=24
    group by 1,2,3) b
on a.segment=b.segment
and a.signup_month=b.signup_month
and a.increments_from_signup=b.increments_from_signup;

--------------net revenue per trip day of activation trip---------
drop table if exists #net_revenue_of_activation_trip;
select a.segment,
       a.signup_month,
       sum(b.gaap_net_revenue::float) as gaap_net_revenue,
       sum(b.paid_days::float) as paid_days,
       sum(b.gaap_net_revenue::float)/sum(b.paid_days::float) as net_revenue_per_day
into #net_revenue_of_activation_trip
from
    (select driver_id,
           segment,
           date_trunc('month',dateadd('month',2,prediction_date)) as signup_month
    from #ltr
    group by 1,2,3) a
left join
    (select * from
        (select rs.driver_id,
           rs.reservation_id,
           d.gaap_net_revenue,
           e.paid_days,
           row_number() over (partition by rs.driver_id order by rs.trip_start_ts) as rank1
            FROM analytics.reservation_summary rs
              inner join (select distinct driver_id from #ltr) b
              on rs.driver_id=b.driver_id
            left join finance.reservation_profit_summary_staging d
                   on rs.reservation_id=d.reservation_id
            left join analytics.reservation_dimensions e
                   on rs.reservation_id=e.reservation_id
             WHERE 1=1
            and rs.is_ever_booked=1 and rs.current_status not in (2,11)
            and rs.trip_start_ts is not null)
    where rank1=1) b
on a.driver_id=b.driver_id
group by 1,2
order by 1,2;

------------num_signups_used_from_each_model--------------
drop table if exists #num_signups_used_from_prediction;
select a.*,
    b.num_signups_from_activation
into #num_signups_used_from_prediction
from (select segment,
        date_trunc('month',dateadd('month',2,prediction_date)) as estimation_month,
        count(distinct driver_id) as num_signups_from_ltr
    from #ltr
    group by 1,2) as a 
    left join (
        select segment,
            date_trunc('month',dateadd('month',2,prediction_date)) as estimation_month,
            count(distinct driver_id) as num_signups_from_activation
        from #activation
        group by 1,2
    ) as b on a.segment=b.segment and a.estimation_month=b.estimation_month;

------------paid days per signup----------------
drop table if exists #paid_days_per_signup_ds;
select a.segment,
       a.signup_month,
       a.increments_from_signup,
       a.activations,
       a.trip_month,
       a.ltr_per_signup,
       b.net_revenue_per_day,
       a.ltr_per_signup::float/b.net_revenue_per_day::float as paid_days_per_signup,
       c.num_signups_from_ltr,
       c.num_signups_from_activation
into #paid_days_per_signup_ds
from #ds_final a
left join #net_revenue_of_activation_trip b
on a.segment=b.segment and a.signup_month=b.signup_month
left join #num_signups_used_from_prediction c
on a.segment=c.segment and a.signup_month=c.estimation_month
order by 1,2,3;

DROP TABLE IF EXISTS #ds_curve_by_signup_month_cps;
WITH pcp_denom AS (
    SELECT *,
           LAG(paid_days_per_signup) OVER (PARTITION BY segment,signup_month ORDER BY increments_from_signup) AS previous_paid_days_per_signup
    FROM #paid_days_per_signup_ds
)

select * INTO #ds_curve_by_signup_month_cps from
(SELECT segment,
       signup_month,
       increments_from_signup,
       num_signups_from_ltr,
       num_signups_from_activation,
       activations,
       sum(activations) over (partition by segment,signup_month order by increments_from_signup ROWS UNBOUNDED PRECEDING) as accumulated_activations,
       paid_days_per_signup,
       previous_paid_days_per_signup,
       CASE WHEN increments_from_signup=1 THEN 1 ELSE paid_days_per_signup::FLOAT /previous_paid_days_per_signup::FLOAT END AS ds_curve
FROM pcp_denom)
;

--adjust seasonal factor
DROP TABLE IF EXISTS #actual_2_increments;

select *,
       paid_days_per_signup_original*seasonal_index as paid_days_per_signup
into #actual_2_increments
from
    (select segment,
           signup_month,
           increments_from_signup,
           signups::float as num_signups_actual_by_increment,
           paid_days::float/signups::float as paid_days_per_signup_original,
           case when date_part('month',signup_month)=1 then 0.93
                when date_part('month',signup_month)=2 then 0.92
                when date_part('month',signup_month)=3 then 1.00
                when date_part('month',signup_month)=4 then 1.07
                when date_part('month',signup_month)=5 then 1.07
                when date_part('month',signup_month)=6 then 1.02
                when date_part('month',signup_month)=7 then 1.06
                when date_part('month',signup_month)=8 then 1.1
                when date_part('month',signup_month)=9 then 1.06
                when date_part('month',signup_month)=10 then 0.95
                when date_part('month',signup_month)=11 then 0.92
                when date_part('month',signup_month)=12 then 0.93
           else null end as seasonal_index
         from #historical_paid_days_per_signup
        -- Take the actuals corresponding to 3 months ago to predict for a given signup month
         where country='US' and increments_from_signup<=2)
;

--need to update join on signup_month, already did
DROP TABLE IF EXISTS #demand_paid_days_cps;
SELECT b.segment,
       b.signup_month,
       b.increments_from_signup,
       b.num_signups_from_ltr,
       b.num_signups_from_activation,
       b.activations,
       b.accumulated_activations,
       b.paid_days_per_signup,
       b.ds_curve,
       a.num_signups_actual_by_increment,
       a.paid_days_per_signup*1 AS projected_paid_days
INTO #demand_paid_days_cps
FROM #actual_2_increments a
         RIGHT JOIN (SELECT * FROM #ds_curve_by_signup_month_cps
                    WHERE increments_from_signup<=12
                    and signup_month<>'2023-06-01') b --no ltr model for June 2023
              ON  a.segment=b.segment
              AND a.signup_month = dateadd('month',-3,b.signup_month) --pay attention
              AND a.increments_from_signup = b.increments_from_signup;

select a.channels,
    b.*
from (
    select channels,
        case when channels in ('Kayak_Desktop', 'Kayak_Desktop_Carousel', 'Kayak_Afterclick', 'Kayak_Desktop_Compare', 'Kayak_Desktop_Front_Door') then 'Kayak_Desktop_Ad'
            when channels in ('Kayak_Mobile_Carousel', 'Kayak_Mobile', 'Kayak_Mobile_Front_Door') then 'Kayak_Mobile_Ad'
            when channels in ('Facebook/IG_App','Apple') then 'all_app'
            when channels in ('Facebook/IG_Web', 'Mediaalpha','Expedia', 'Reddit') then 'all web'
            else channels end as segment
    from (select distinct channels from #temp_signup_base)
) a
left join #demand_paid_days_cps b
on a.segment = b.segment 
where channels in ('Apple', 'Apple_Brand', 'Google_Desktop','Google_Desktop_Brand',
            'Google_Mobile','Google_Mobile_Brand','Kayak_Desktop', 'Kayak_Desktop_Core', 
            'Kayak_Mobile_Core','Mediaalpha','Expedia','Microsoft_Desktop',
            'Microsoft_Desktop_Brand', 'Reddit', 'Moloco', 'Kayak_Desktop_Compare',
            'Google_Pmax','Kayak_Desktop_Carousel','Kayak_Mobile_Carousel',
            'Kayak_Afterclick', 'Facebook/IG_App', 'Facebook/IG_Web')
