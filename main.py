from asset import stock

stock_value = stock.Value("2006-01-01", "2023-06-01")
stock_value.create_factor_data()

stock_growth = stock.Growth("2006-01-01", "2023-06-01")
stock_growth.create_factor_data()

stock_size = stock.Size("2006-01-01", "2023-06-01")
stock_size.create_factor_data()