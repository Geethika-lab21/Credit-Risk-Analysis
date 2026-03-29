DROP TABLE IF EXISTS loans;

CREATE TABLE loans (
    loan_amnt FLOAT,
    int_rate FLOAT,
    installment FLOAT,
    annual_inc FLOAT,
    dti FLOAT,
    revol_util FLOAT,
    revol_bal FLOAT,
    open_acc INT,
    total_acc INT,
    loan_status TEXT
);