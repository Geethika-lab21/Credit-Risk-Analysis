-- Default count
SELECT loan_status, COUNT(*) 
FROM loans 
GROUP BY loan_status;

-- Avg income
SELECT loan_status, AVG(annual_inc)
FROM loans
GROUP BY loan_status;

-- High risk users
SELECT *
FROM loans
WHERE dti > 20 AND int_rate > 15;