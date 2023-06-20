import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Chemin du dossier contenant les fichiers CSV
dossier = 'vendor/benchmarks/ant_complete'

# Chemin du dossier où les graphiques seront enregistrés
save_folder = 'vendor/reports'

# Listes pour stocker les données extraites
nb_nodes = []
costs = []
execution_time = []
nb_ants = []
alpha = []
beta = []
evaporation = []
nb_iterations = []
best_costs = []

# Parcourir tous les fichiers CSV dans le dossier
for filename in os.listdir(dossier):
    file_path = os.path.join(dossier, filename)
    if filename.endswith('.csv'):
        # Vérifier si le fichier est vide
        if os.path.getsize(file_path) == 0:
            print(f"Le fichier {filename} est vide. Ignoré.")
            continue

        # Charger les données CSV
        try:
            data = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            print(f"Le fichier {filename} est vide ou n'a pas pu être lu. Ignoré.")
            continue

        # Extraire et stocker les données requises
        if 'nb_nodes' in data.columns:
            nb_nodes.extend(data['nb_nodes'])

        if 'runtime (ms)' in data.columns:
            execution_time.extend(data['runtime (ms)'])

        if 'nb_ants' in data.columns:
            nb_ants.extend(data['nb_ants'])

        if 'alpha' in data.columns:
            alpha.extend(data['alpha'])

        if 'beta' in data.columns:
            beta.extend(data['beta'])

        if 'evap' in data.columns:
            evaporation.extend(data['evap'])

        if 'nb_iter' in data.columns:
            nb_iterations.extend(data['nb_iter'])

        if 'cost' in data.columns:
            best_costs.extend(data['cost'])

# Graphique 1 : Temps d'exécution vs Nombre de nœuds
plt.figure(figsize=(10, 5))
plt.scatter(nb_nodes, execution_time, label='Data points')
m, b = np.polyfit(nb_nodes, execution_time, 1)
plt.plot(nb_nodes, m * np.array(nb_nodes) + b, color='red', label='Average line')
plt.xlabel('Nombre de nœuds')
plt.ylabel('Temps d\'exécution (ms)')
plt.title('Temps d\'exécution en fonction du nombre de nœuds')
plt.legend()
plt.grid(True)
plt.savefig(f'{save_folder}/temps_execution_vs_noeuds.png')

# Graphique 2 : Cost vs Nombre de nœuds avec best_cost comme couleur
plt.figure(figsize=(10, 5))
plt.scatter(nb_nodes, best_costs, c=nb_ants, cmap='viridis', label='Data points')
m, b = np.polyfit(nb_nodes, best_costs, 1)
plt.plot(nb_nodes, m * np.array(nb_nodes) + b, color='red', label='Average line')
plt.colorbar(label='Best Cost')
plt.xlabel('Nombre de nœuds')
plt.ylabel('Coût')
plt.title('Coût en fonction du nombre de nœuds')
plt.legend()
plt.grid(True)
plt.savefig(f'{save_folder}/cout_vs_noeuds.png')

# Graphique 3 : Temps d'exécution vs Nombre de fourmis avec best_cost comme couleur
plt.figure(figsize=(10, 5))
plt.scatter(nb_ants, execution_time, c=best_costs, cmap='viridis', label='Data points')
plt.colorbar(label='Best Cost')
plt.xlabel('Nombre de fourmis')
plt.ylabel('Temps d\'exécution (ms)')
plt.title('Temps d\'exécution en fonction du nombre de fourmis')
plt.legend()
plt.grid(True)
plt.show()

# Graphique 3 : Temps d'exécution vs Nombre de fourmis avec best_cost comme couleur
plt.figure(figsize=(10, 5))
plt.scatter(nb_ants, execution_time, c=best_costs, cmap='viridis', label='Data points')
m, b = np.polyfit(nb_ants, execution_time, 1)
plt.plot(nb_ants, m * np.array(nb_ants) + b, color='red', label='Average line')
plt.colorbar(label='Best Cost')
plt.xlabel('Nombre de fourmis')
plt.ylabel('Temps d\'exécution (ms)')
plt.title('Temps d\'exécution en fonction du nombre de fourmis')
plt.legend()
plt.grid(True)
plt.savefig(f'{save_folder}/temps_execution_vs_fourmis.png')

# Graphique 5 : Temps d'exécution vs Beta
plt.figure(figsize=(10, 5))
plt.scatter(beta, execution_time, c=best_costs, cmap='viridis', label='Data points')
m, b = np.polyfit(beta, execution_time, 1)
plt.plot(beta, m * np.array(beta) + b, color='red', label='Average line')
plt.colorbar(label='Best Cost')
plt.xlabel('Beta')
plt.ylabel('Temps d\'exécution (ms)')
plt.title('Temps d\'exécution en fonction de beta')
plt.legend()
plt.grid(True)
plt.savefig(f'{save_folder}/temps_execution_vs_beta.png')

# Graphique 6 : Temps d'exécution vs Evaporation
plt.figure(figsize=(10, 5))
plt.scatter(evaporation, execution_time, c=best_costs, cmap='viridis', label='Data points')
m, b = np.polyfit(evaporation, execution_time, 1)
plt.plot(evaporation, m * np.array(evaporation) + b, color='red', label='Average line')
plt.colorbar(label='Best Cost')
plt.xlabel('Evaporation')
plt.ylabel('Temps d\'exécution (ms)')
plt.title('Temps d\'exécution en fonction de l\'evaporation')
plt.legend()
plt.grid(True)
plt.savefig(f'{save_folder}/temps_execution_vs_evaporation.png')

# Graphique 7 : Temps d'exécution vs Nombre d'itérations
plt.figure(figsize=(10, 5))
plt.scatter(nb_iterations, execution_time, c=best_costs, cmap='viridis', label='Data points')
m, b = np.polyfit(nb_iterations, execution_time, 1)
plt.plot(nb_iterations, m * np.array(nb_iterations) + b, color='red', label='Average line')
plt.colorbar(label='Best Cost')
plt.xlabel('Nombre d\'itérations')
plt.ylabel('Temps d\'exécution (ms)')
plt.title('Temps d\'exécution en fonction du nombre d\'itérations')
plt.legend()
plt.grid(True)
plt.savefig(f'{save_folder}/temps_execution_vs_iterations.png')

# Graphique 8 : Distance vs Nombre d'itérations
plt.figure(figsize=(10, 5))
plt.scatter(execution_time, best_costs, c=nb_iterations, cmap='viridis', label='Data points')
m, b = np.polyfit(execution_time, best_costs, 1)
plt.plot(execution_time, m * np.array(execution_time) + b, color='red', label='Average line')
plt.colorbar(label='Nombre d\'itérations')
plt.xlabel('Temps d\'exécution (ms)')
plt.ylabel('Distance')
plt.title('Distance en fonction du temps d\'exécution')
plt.legend()
plt.grid(True)
plt.savefig(f'{save_folder}/distance_vs_temps_execution.png')