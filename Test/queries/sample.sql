SELECT u.username,
       u.city,
       o.id AS order_id,
       o.status,
       o.total_amount,
       o.created_at
FROM users u
         JOIN orders o ON u.id = o.user_id
WHERE o.status = 'paid'
  AND u.city IN ('Shanghai', 'Beijing')
  AND o.total_amount > 5000;
