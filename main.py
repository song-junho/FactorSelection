from asset import stock

stock_value = stock.Value("2006-01-01", "2023-03-31")
stock_value.create_factor_data()

# stock_growth = stock.Growth("2006-01-01", "2023-03-31")
# stock_growth.create_factor_data()