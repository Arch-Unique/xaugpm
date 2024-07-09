from predict import predict_gold_price

# Predict the next hour using the last 10 days of data
next_hour_prices = predict_gold_price('path/to/last_10_days.csv')
print(next_hour_prices)