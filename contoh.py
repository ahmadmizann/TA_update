from scipy.stats import poisson

lambda_a = 0.3333  # Rata-rata gol Tim Indonesia per pertandingan
lambda_b = 2.6666  # Rata-rata gol Tim Australia per pertandingan

prob_a = 0.0
prob_b = 0.0

for x_a in range(7):  # x_a adalah jumlah gol yang dicetak oleh Tim Indonesia
    for x_b in range(7):  # x_b adalah jumlah gol yang dicetak oleh Tim Australia
        # Menghitung probabilitas bahwa Tim Indonesia mencetak x_a gol dan Tim Australia mencetak x_b gol
        p_total = poisson.pmf(x_a, lambda_a) * poisson.pmf(x_b, lambda_b)
        
        # Memeriksa apakah Tim Indonesia mencetak lebih banyak gol daripada Tim Australia
        if x_a > x_b:
            prob_a += p_total
        elif x_a < x_b:
            prob_b += p_total

# Normalisasi probabilitas
total_prob = prob_a + prob_b
prob_a /= total_prob
prob_b /= total_prob

print("Probabilitas bahwa Tim Indonesia mencetak lebih banyak gol daripada Tim Australia:", prob_b)
print("Probabilitas bahwa Tim Australia mencetak lebih banyak gol daripada Tim Indonesia:", prob_a)
