from asset import stock

start_date = "2006-01-01"
end_date   = "2023-08-01"

if __name__ == "__main__":

    stock.Value(start_date, end_date).create_factor_data()
    stock.Growth(start_date, end_date).create_factor_data()
    stock.Size(start_date, end_date).create_factor_data()
    stock.Quality(start_date, end_date).create_factor_data()
    stock.Momentum(start_date, end_date).create_factor_data()
