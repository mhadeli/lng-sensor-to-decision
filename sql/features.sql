-- One row per machine at a six-hour decision point. All input windows end before t.
WITH errors_by_hour AS (
    SELECT machineID, datetime,
           count(*) FILTER (WHERE errorID = 'error1') AS error1,
           count(*) FILTER (WHERE errorID = 'error2') AS error2,
           count(*) FILTER (WHERE errorID = 'error3') AS error3,
           count(*) FILTER (WHERE errorID = 'error4') AS error4,
           count(*) FILTER (WHERE errorID = 'error5') AS error5
    FROM errors GROUP BY machineID, datetime
), history AS (
    SELECT t.machineID, t.datetime,
           count(*) OVER w AS history_hours,
           avg(volt) OVER w AS volt_mean, stddev_pop(volt) OVER w AS volt_std,
           avg(rotate) OVER w AS rotate_mean, stddev_pop(rotate) OVER w AS rotate_std,
           avg(pressure) OVER w AS pressure_mean, stddev_pop(pressure) OVER w AS pressure_std,
           avg(vibration) OVER w AS vibration_mean, stddev_pop(vibration) OVER w AS vibration_std,
           sum(coalesce(error1, 0)) OVER w AS error1_count,
           sum(coalesce(error2, 0)) OVER w AS error2_count,
           sum(coalesce(error3, 0)) OVER w AS error3_count,
           sum(coalesce(error4, 0)) OVER w AS error4_count,
           sum(coalesce(error5, 0)) OVER w AS error5_count
    FROM telemetry t LEFT JOIN errors_by_hour e USING (machineID, datetime)
    WINDOW w AS (PARTITION BY t.machineID ORDER BY t.datetime
        RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING)
), decisions AS (
    SELECT * FROM history WHERE history_hours = 24 AND hour(datetime) % 6 = 0
)
SELECT t.*, m.age,
       (m.model = 'model1')::INTEGER AS model1,
       (m.model = 'model2')::INTEGER AS model2,
       (m.model = 'model3')::INTEGER AS model3,
       (m.model = 'model4')::INTEGER AS model4,
       coalesce(date_diff('hour', c1.datetime, t.datetime) / 24.0, 365.0) AS comp1_days,
       coalesce(date_diff('hour', c2.datetime, t.datetime) / 24.0, 365.0) AS comp2_days,
       coalesce(date_diff('hour', c3.datetime, t.datetime) / 24.0, 365.0) AS comp3_days,
       coalesce(date_diff('hour', c4.datetime, t.datetime) / 24.0, 365.0) AS comp4_days,
       (c1.datetime IS NULL)::INTEGER AS comp1_unknown,
       (c2.datetime IS NULL)::INTEGER AS comp2_unknown,
       (c3.datetime IS NULL)::INTEGER AS comp3_unknown,
       (c4.datetime IS NULL)::INTEGER AS comp4_unknown
FROM decisions t
JOIN machines m USING (machineID)
ASOF LEFT JOIN (SELECT * FROM maintenance WHERE comp = 'comp1') c1
    ON t.machineID = c1.machineID AND t.datetime > c1.datetime
ASOF LEFT JOIN (SELECT * FROM maintenance WHERE comp = 'comp2') c2
    ON t.machineID = c2.machineID AND t.datetime > c2.datetime
ASOF LEFT JOIN (SELECT * FROM maintenance WHERE comp = 'comp3') c3
    ON t.machineID = c3.machineID AND t.datetime > c3.datetime
ASOF LEFT JOIN (SELECT * FROM maintenance WHERE comp = 'comp4') c4
    ON t.machineID = c4.machineID AND t.datetime > c4.datetime
ORDER BY t.machineID, t.datetime
