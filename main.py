import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import unittest
import math
from sqlalchemy.sql import text

# Fehlerbehandlung für Dateiladen
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    ideal_df = pd.read_csv('ideal.csv')
except FileNotFoundError as e:
    print(f"Fehler: Datei nicht gefunden - {e}")
    exit(1)

# 2. SQLite-Datenbank erstellen
try:
    engine = create_engine('sqlite:///assignment.db')
    train_df.to_sql('training_data', engine, if_exists='replace', index=False)
    ideal_df.to_sql('ideal_functions', engine, if_exists='replace', index=False)
except Exception as e:
    print(f"Fehler beim Erstellen der Datenbank: {e}")
    exit(1)

# 3. Beste ideale Funktionen auswählen (Least-Square)
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

best_functions = {}
max_deviations = {}

for y_col in ['y1', 'y2', 'y3', 'y4']:
    min_mse = float('inf')
    best_func = None
    y_true = train_df[y_col].values
    for ideal_col in [f'y{i}' for i in range(1, 51)]:
        y_pred = ideal_df[ideal_col].values
        mse = calculate_mse(y_true, y_pred)
        if mse < min_mse:
            min_mse = mse
            best_func = ideal_col
    best_functions[y_col] = best_func
    max_deviation = np.max(np.abs(y_true - ideal_df[best_func].values))
    max_deviations[y_col] = max_deviation

print("Ausgewählte Funktionen:", best_functions)
print("Maximale Abweichungen:", max_deviations)

# 4. Testdaten zuordnen und Ergebnisse speichern
results = []
sqrt_2 = math.sqrt(2)

for _, row in test_df.iterrows():
    x, y = row['x'], row['y']
    assigned_func = None
    min_deviation = float('inf')
    
    for y_col, ideal_func in best_functions.items():
        # Interpolation für x-Werte, die nicht exakt übereinstimmen
        if x in ideal_df['x'].values:
            ideal_y = ideal_df[ideal_df['x'] == x][ideal_func].values[0]
        else:
            ideal_y = np.interp(x, ideal_df['x'], ideal_df[ideal_func])
        deviation = abs(y - ideal_y)
        max_allowed_deviation = max_deviations[y_col] * sqrt_2
        if deviation <= max_allowed_deviation and deviation < min_deviation:
            min_deviation = deviation
            assigned_func = ideal_func
    
    if assigned_func:
        results.append({
            'x': x,
            'y': y,
            'chosen_function': assigned_func,
            'deviation': min_deviation
        })

# Ergebnisse in die Datenbank speichern
results_df = pd.DataFrame(results)
try:
    results_df.to_sql('test_results', engine, if_exists='replace', index=False)
except Exception as e:
    print(f"Fehler beim Speichern der Ergebnisse: {e}")
    exit(1)

# Datenbankinhalt ausgeben
with engine.connect() as conn:
    print("\nTrainingsdaten (erste 5 Zeilen):")
    print(pd.read_sql("SELECT * FROM training_data LIMIT 5", conn))
    print("\nIdeale Funktionen (erste 5 Zeilen):")
    print(pd.read_sql("SELECT * FROM ideal_functions LIMIT 5", conn))
    print("\nTest-Ergebnisse (erste 5 Zeilen):")
    print(pd.read_sql("SELECT * FROM test_results LIMIT 5", conn))

# 5. Visualisierung
plt.figure(figsize=(12, 8))
for y_col in ['y1', 'y2', 'y3', 'y4']:
    plt.plot(train_df['x'], train_df[y_col], label=f'Training {y_col}', linestyle='--')
for y_col, ideal_func in best_functions.items():
    plt.plot(ideal_df['x'], ideal_df[ideal_func], label=f'Ideal {ideal_func}', alpha=0.7)
plt.scatter(test_df['x'], test_df['y'], color='red', label='Testdaten', zorder=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trainingsdaten, Testdaten und ausgewählte ideale Funktionen')
plt.legend()
plt.grid(True)
plt.savefig('visualization.png')
plt.close()
print("Visualisierung gespeichert als 'visualization.png'")

# 6. Unit-Tests
class TestAssignment(unittest.TestCase):
    def test_mse_calculation(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.1, 3.1])
        mse = calculate_mse(y_true, y_pred)
        self.assertAlmostEqual(mse, 0.01, places=4)
    
    def test_database_tables(self):
        with engine.connect() as conn:
            tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()
            table_names = [t[0] for t in tables]
            self.assertIn('training_data', table_names)
            self.assertIn('ideal_functions', table_names)
            self.assertIn('test_results', table_names)
    
    def test_function_assignment(self):
        self.assertTrue(len(results_df) > 0, "Es wurden keine Testpunkte zugeordnet")
        self.assertTrue(all(results_df['deviation'] >= 0), "Abweichungen sollten nicht negativ sein")

if __name__ == '__main__':
    print("Starte Unit-Tests...")
    unittest.main(argv=[''], verbosity=2, exit=False)