-- 为了方便重复执行，先删表
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS users;

-- 用户表
CREATE TABLE users
(
    id         SERIAL PRIMARY KEY,
    username   VARCHAR(50)  NOT NULL UNIQUE,
    email      VARCHAR(100) NOT NULL UNIQUE,
    age        INT,
    city       VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 商品表
CREATE TABLE products
(
    id         SERIAL PRIMARY KEY,
    name       VARCHAR(100)   NOT NULL,
    category   VARCHAR(50),
    price      NUMERIC(10, 2) NOT NULL,
    stock      INT            NOT NULL DEFAULT 0,
    created_at TIMESTAMP               DEFAULT CURRENT_TIMESTAMP
);

-- 订单表
CREATE TABLE orders
(
    id           SERIAL PRIMARY KEY,
    user_id      INT            NOT NULL REFERENCES users (id),
    status       VARCHAR(20)    NOT NULL,
    total_amount NUMERIC(10, 2) NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 订单明细表
CREATE TABLE order_items
(
    id         SERIAL PRIMARY KEY,
    order_id   INT            NOT NULL REFERENCES orders (id) ON DELETE CASCADE,
    product_id INT            NOT NULL REFERENCES products (id),
    quantity   INT            NOT NULL,
    unit_price NUMERIC(10, 2) NOT NULL
);

-- 插入用户数据
INSERT INTO users (username, email, age, city, created_at)
VALUES ('alice', 'alice@example.com', 25, 'Shanghai', '2026-05-01 10:10:00'),
       ('bob', 'bob@example.com', 31, 'Beijing', '2026-05-02 11:20:00'),
       ('charlie', 'charlie@example.com', 28, 'Shenzhen', '2026-05-03 09:30:00'),
       ('diana', 'diana@example.com', 35, 'Guangzhou', '2026-05-04 14:40:00'),
       ('eric', 'eric@example.com', 22, 'Hangzhou', '2026-05-05 16:00:00'),
       ('fiona', 'fiona@example.com', 29, 'Chengdu', '2026-05-06 18:15:00'),
       ('george', 'george@example.com', 40, 'Nanjing', '2026-05-07 08:45:00'),
       ('helen', 'helen@example.com', 27, 'Suzhou', '2026-05-08 12:00:00');

-- 插入商品数据
INSERT INTO products (name, category, price, stock, created_at)
VALUES ('MacBook Pro 14', 'Laptop', 15999.00, 20, '2026-04-20 09:00:00'),
       ('ThinkPad X1 Carbon', 'Laptop', 12999.00, 15, '2026-04-21 09:00:00'),
       ('iPhone 16', 'Phone', 6999.00, 50, '2026-04-22 09:00:00'),
       ('Galaxy S26', 'Phone', 6499.00, 45, '2026-04-23 09:00:00'),
       ('AirPods Pro', 'Audio', 1899.00, 100, '2026-04-24 09:00:00'),
       ('Sony WH-1000XM6', 'Audio', 2999.00, 35, '2026-04-25 09:00:00'),
       ('iPad Air', 'Tablet', 4799.00, 30, '2026-04-26 09:00:00'),
       ('Logitech MX Master 3S', 'Accessory', 699.00, 80, '2026-04-27 09:00:00');

-- 插入订单数据
INSERT INTO orders (user_id, status, total_amount, created_at)
VALUES (1, 'paid', 17898.00, '2026-05-10 10:00:00'),
       (2, 'paid', 6999.00, '2026-05-10 11:30:00'),
       (3, 'pending', 3698.00, '2026-05-11 09:20:00'),
       (4, 'cancelled', 12999.00, '2026-05-11 15:45:00'),
       (5, 'paid', 5498.00, '2026-05-12 13:10:00'),
       (6, 'paid', 2298.00, '2026-05-13 17:35:00'),
       (7, 'pending', 15999.00, '2026-05-14 19:00:00'),
       (8, 'paid', 9498.00, '2026-05-15 20:20:00');

-- 插入订单明细数据
INSERT INTO order_items (order_id, product_id, quantity, unit_price)
VALUES (1, 1, 1, 15999.00),
       (1, 5, 1, 1899.00),

       (2, 3, 1, 6999.00),

       (3, 5, 1, 1899.00),
       (3, 8, 1, 699.00),
       (3, 8, 1, 699.00),
       (3, 8, 1, 401.00),

       (4, 2, 1, 12999.00),

       (5, 7, 1, 4799.00),
       (5, 8, 1, 699.00),

       (6, 5, 1, 1899.00),
       (6, 8, 1, 399.00),

       (7, 1, 1, 15999.00),

       (8, 4, 1, 6499.00),
       (8, 6, 1, 2999.00);

-- 简单查询验证
SELECT 'users' AS table_name, COUNT(*) AS row_count
FROM users
UNION ALL
SELECT 'products', COUNT(*)
FROM products
UNION ALL
SELECT 'orders', COUNT(*)
FROM orders
UNION ALL
SELECT 'order_items', COUNT(*)
FROM order_items;
