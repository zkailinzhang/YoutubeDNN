self.action_table = """
    
    SELECT t1.distinct_id, CONCAT(t1.item_id, ',', t1.item_type) as item_id, t1.event_time, t3.category {item_infos_sql}
    FROM (
        SELECT distinct_id, {item_id_col} AS item_id, regexp_replace({item_type_col},'-','') as item_type,
                unix_timestamp(MAX(time)) AS event_time
        FROM events
        WHERE event = '{event_name}'
            AND date >= '{start_date}'
            AND date <= '{end_date}'
            AND time >= '{start_time}'
            AND time < '{end_time}'
            AND distinct_id IS NOT NULL
            AND distinct_id <> ''
            {evt_other_condition}
        GROUP BY distinct_id, item_id, item_type
    ) t1
    JOIN (
        SELECT {item_id_col} AS item_id, regexp_replace({item_type_col},'-','') as item_type
        FROM events
        WHERE event = '{event_name}'
            AND date >= '{start_date}'
            AND date <= '{end_date}'
            AND time >= '{start_time}'
            AND time < '{end_time}'
            AND distinct_id IS NOT NULL
            AND distinct_id <> ''
            {evt_other_condition}
        GROUP BY item_id, item_type
        HAVING COUNT(DISTINCT(distinct_id)) >= {min_item_frequency}
    ) t2
    ON t1.item_id = t2.item_id AND t1.item_type = t2.item_type
    JOIN (
        SELECT item_id, item_type, {category_col} as category {item_infos_sql}
        FROM {items_table}
        
    ) t3
    ON t1.item_id = t3.item_id AND t1.item_type = t3.item_type
    WHERE t1.item_id IS NOT NULL
        AND t1.item_id <> ''
"""



SELECT t1.distinct_id, CONCAT(t1.item_id, ',', t1.item_type) as item_id, t1.event_time, t3.category ,t3.tags
        FROM (
            SELECT distinct_id, substanceid AS item_id, regexp_replace(contenttype,'-','') as item_type,
                    unix_timestamp(MAX(time)) AS event_time
            FROM events
            WHERE event = 'contentClick'
                AND date >= '2019-06-01'
                AND date <= '2020-06-01'
                AND distinct_id IS NOT NULL
                AND distinct_id <> ''
                and length(substanceid)>0 and ((contenttype in ('PS','TX-PS') and firstlevelprogramtype!='综艺') or (contenttype in ('PS','TX-CS') and firstlevelprogramtype='综艺'))
            GROUP BY distinct_id, item_id, item_type
        ) t1
        JOIN (
            SELECT substanceid AS item_id, regexp_replace(contenttype,'-','') as item_type
            FROM events
            WHERE event = 'contentClick'
                AND date >= '2019-06-01'
                AND date <= '2020-06-01'
                AND distinct_id IS NOT NULL
                AND distinct_id <> ''
                and length(substanceid)>0 and ((contenttype in ('PS','TX-PS') and firstlevelprogramtype!='综艺') or (contenttype in ('PS','TX-CS') and firstlevelprogramtype='综艺'))
            GROUP BY item_id, item_type
            HAVING COUNT(DISTINCT(distinct_id)) >= 3
        ) t2
        ON t1.item_id = t2.item_id AND t1.item_type = t2.item_type
        JOIN (
            SELECT item_id, item_type, category as category ,tags
            FROM items
            where state=0 and category is not null and ((item_type in ('PS', 'TXPS') and category !='综艺') or (item_type in ('PS', 'TXCS') and category ='综艺'))
            and item_id not in (select item_id from items
                                where 
                                ((category = '综艺' and (category_second like '%片段%' or category_second like '%策划%'
      or category_second like '%花絮%')) or (category = '电影' and (category_second like '%电影特辑%' )) or
      (category = '电视剧' and (category_second like '%创意剪辑%' or category_second like '%片花%')))
                                 )
        ) t3
        ON t1.item_id = t3.item_id AND t1.item_type = t3.item_type
        WHERE t1.item_id IS NOT NULL