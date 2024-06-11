import pandas as pd
import numpy as np
import multiprocessing as mp
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import scipy.stats as stats

def simulate_sales_batch(num_days, seed, start_day):
    np.random.seed(seed)
    climate_data = {
        'clima': ['caluroso', 'templado', 'frio', 'helado'],
        'probabilidad': [0.5, 0.25, 0.15, 0.1]
    }

    sales_data = {
        'clima': ['caluroso', 'caluroso', 'caluroso', 'caluroso', 'templado', 'templado', 'templado', 'templado', 'frio', 'frio', 'frio', 'frio', 'helado', 'helado', 'helado', 'helado'],
        'litros_vendidos_dia': [50, 100, 200, 300, 40, 50, 100, 200, 10, 20, 50, 100, 0, 5, 10, 20],
        'probabilidad': [0.1, 0.3, 0.4, 0.2, 0.1, 0.1, 0.3, 0.5, 0.05, 0.7, 0.2, 0.05, 0.05, 0.1, 0.7, 0.15]
    }

    daily_cost = 1000
    price_per_liter = 50

    df_climate = pd.DataFrame(climate_data)
    df_sales = pd.DataFrame(sales_data)

    df_climate['probabilidad_acumulada'] = df_climate['probabilidad'].cumsum()
    df_climate['V_min'] = df_climate['probabilidad_acumulada'].shift(1).fillna(0)
    df_climate['V_max'] = df_climate['probabilidad_acumulada']

    df_sales['probabilidad_acumulada'] = df_sales.groupby('clima')['probabilidad'].cumsum()
    df_sales['V_min'] = df_sales.groupby('clima')['probabilidad_acumulada'].shift(1).fillna(0)
    df_sales['V_max'] = df_sales['probabilidad_acumulada']

    sim_data = []

    for day in range(start_day, start_day + num_days):
        NA = np.random.rand()
        clima_row = df_climate[(df_climate['V_min'] <= NA) & (df_climate['V_max'] > NA)].iloc[0]
        climate = clima_row['clima']
        NA2 = np.random.rand()
        sales_rows = df_sales[(df_sales['clima'] == climate) & (df_sales['V_min'] <= NA2) & (df_sales['V_max'] > NA2)]
        if not sales_rows.empty:
            sales_row = sales_rows.iloc[0]
            litros_vendidos = sales_row['litros_vendidos_dia']
        else:
            litros_vendidos = np.nan  # This ensures it is clearly seen if there is an issue
        ingreso = litros_vendidos * price_per_liter
        utilidad_diaria = ingreso - daily_cost
        sim_data.append([day, NA, climate, NA2, litros_vendidos, ingreso, utilidad_diaria])

    df_sim = pd.DataFrame(sim_data, columns=['nro_dias', 'NA', 'X', 'NA2', 'litros_vendidos', 'Ingreso', 'Utilidad_diaria'])
    df_sim['Promedio_Movil'] = df_sim['Utilidad_diaria'].expanding().mean()
    df_sim['nro_dias'] = df_sim['nro_dias'].astype(int)

    return df_sim

def run_simulation():
    try:
        num_days = int(entry.get())
        if num_days <= 0:
            raise ValueError
    except ValueError:
        result_label.config(text="Please enter a valid positive integer.")
        return

    # Split the workload into batches
    num_batches = mp.cpu_count()
    days_per_batch = num_days // num_batches
    extra_days = num_days % num_batches

    seeds = np.random.randint(0, 100000, size=num_batches)
    batches = [days_per_batch] * num_batches
    for i in range(extra_days):
        batches[i] += 1

    pool = mp.Pool(processes=num_batches)
    results = pool.starmap(simulate_sales_batch, [(batches[i], seeds[i], sum(batches[:i]) + 1) for i in range(num_batches)])
    pool.close()
    pool.join()

    global final_result
    final_result = pd.concat(results).reset_index(drop=True)
    result_text.delete('1.0', tk.END)
    result_text.insert(tk.END, final_result.to_string(index=False))
    
    std_dev = final_result['Promedio_Movil'].std()
    std_dev_label.config(text=f"Standard Deviation: {std_dev:.2f}")

    simulate_replicas_button.config(state=tk.NORMAL)  # Enable the button after running the simulation

def calculate_num_runs():
    try:
        alpha = float(alpha_entry.get())
        epsilon = float(epsilon_entry.get())
        if alpha <= 0 or epsilon <= 0:
            raise ValueError
    except ValueError:
        num_runs_label.config(text="Please enter valid positive numbers for alpha and epsilon.")
        return

    if 'final_result' in globals():
        std_dev = final_result['Promedio_Movil'].std()
        num_runs = (1 / alpha) * ((std_dev / epsilon) ** 2)
        num_runs_label.config(text=f"Optimal Number of Runs: {num_runs:.2f}")
    else:
        num_runs_label.config(text="No simulation data available.")

def plot_probability(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    res = stats.probplot(data, dist="norm", plot=ax)
    plt.title('Gráfica de probabilidad de LastColumn')
    plt.xlabel('Teórico')
    plt.ylabel('Datos')
    plt.grid(True)
    
    # Calculate statistics
    mean = np.mean(data)
    std_dev = np.std(data)
    n = len(data)
    ad_stat, critical_values, significance_level = stats.anderson(data, dist='norm')
    p_value = significance_level[2]  # 5% significance level
    
    # Add text box with statistics
    textstr = '\n'.join((
        f'Media: {mean:.4f}',
        f'Desv.Est.: {std_dev:.4f}',
        f'N: {n}',
        f'AD: {ad_stat:.4f}',
        f'Valor p: {p_value:.4f}'))
    
    # Add these stats as a text box in the plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.show()

def run_simulation_and_plot():
    global final_result
    run_simulation()  # Run the simulation to generate the data
    data = final_result['Promedio_Movil'].dropna()  # Use the 'Promedio_Movil' column from the simulation result
    plot_probability(data)

def simulate_replicas():
    def run_replicas():
        try:
            num_replicas = int(replica_entry.get())
            if num_replicas <= 0:
                raise ValueError
        except ValueError:
            replica_result_label.config(text="Please enter a valid positive integer for replicas.")
            return

        replicas_data = []
        num_days = int(entry.get())
        for _ in range(num_replicas):
            result = simulate_sales_batch(num_days, np.random.randint(0, 100000), 1)
            replicas_data.append(result['Promedio_Movil'].reset_index(drop=True))

        replica_results = pd.concat(replicas_data, axis=1)
        replica_text.delete('1.0', tk.END)
        replica_text.insert(tk.END, replica_results.to_string(index=False))

        global final_replicas
        final_replicas = replica_results

        plot_replicas_button.config(state=tk.NORMAL)
        plot_line_button.config(state=tk.NORMAL)

    def plot_replicas():
        global final_replicas
        final_replicas.boxplot()
        plt.title('Boxplot of Replicas')
        plt.ylabel('Promedio Movil')
        plt.show()

    def plot_line():
        global final_replicas
        fig, ax = plt.subplots(figsize=(8, 6))
        for column in final_replicas.columns:
            ax.plot(final_replicas.index, final_replicas[column], marker='o')
        ax.set_title('Line Plot of Replicas')
        ax.set_xlabel('Days')
        ax.set_ylabel('Promedio Movil')
        ax.grid(True)
        plt.show()

    replica_window = tk.Toplevel(app)
    replica_window.title("Simulate Replicas")

    ttk.Label(replica_window, text="Enter number of replicas:").grid(row=0, column=0, padx=5, pady=5)

    replica_entry = ttk.Entry(replica_window, width=10)
    replica_entry.grid(row=0, column=1, padx=5, pady=5)

    replica_run_button = ttk.Button(replica_window, text="Run Replicas", command=run_replicas)
    replica_run_button.grid(row=0, column=2, padx=5, pady=5)

    replica_result_label = ttk.Label(replica_window, text="")
    replica_result_label.grid(row=0, column=3, padx=5, pady=5)

    replica_result_frame = ttk.Frame(replica_window)
    replica_result_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

    replica_text = tk.Text(replica_result_frame, wrap='none', height=20, width=100)
    replica_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    replica_scrollbar_y = ttk.Scrollbar(replica_result_frame, orient='vertical', command=replica_text.yview)
    replica_scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
    replica_text['yscrollcommand'] = replica_scrollbar_y.set

    replica_scrollbar_x = ttk.Scrollbar(replica_result_frame, orient='horizontal', command=replica_text.xview)
    replica_scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
    replica_text['xscrollcommand'] = replica_scrollbar_x.set

    plot_replicas_button = ttk.Button(replica_window, text="Plot Box Plot", command=plot_replicas)
    plot_replicas_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
    plot_replicas_button.config(state=tk.DISABLED)

    plot_line_button = ttk.Button(replica_window, text="Plot Line Plot", command=plot_line)
    plot_line_button.grid(row=2, column=2, columnspan=2, padx=5, pady=5)
    plot_line_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    # Ensure the entry point is protected
    mp.freeze_support()

    app = tk.Tk()
    app.title("Sales Simulation")

    frame = ttk.Frame(app, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    label = ttk.Label(frame, text="Enter number of days:")
    label.grid(row=0, column=0, padx=5, pady=5)

    entry = ttk.Entry(frame, width=10)
    entry.grid(row=0, column=1, padx=5, pady=5)

    button = ttk.Button(frame, text="Run Simulation", command=run_simulation)
    button.grid(row=0, column=2, padx=5, pady=5)

    std_dev_label = ttk.Label(frame, text="")
    std_dev_label.grid(row=0, column=3, padx=5, pady=5)

    alpha_label = ttk.Label(frame, text="Alpha (α):")
    alpha_label.grid(row=1, column=0, padx=5, pady=5)

    alpha_entry = ttk.Entry(frame, width=10)
    alpha_entry.grid(row=1, column=1, padx=5, pady=5)

    epsilon_label = ttk.Label(frame, text="Epsilon (ε):")
    epsilon_label.grid(row=1, column=2, padx=5, pady=5)

    epsilon_entry = ttk.Entry(frame, width=10)
    epsilon_entry.grid(row=1, column=3, padx=5, pady=5)

    num_runs_button = ttk.Button(frame, text="Calculate Num Runs", command=calculate_num_runs)
    num_runs_button.grid(row=1, column=4, padx=5, pady=5)

    num_runs_label = ttk.Label(frame, text="")
    num_runs_label.grid(row=1, column=5, padx=5, pady=5)

    plot_button = ttk.Button(frame, text="Plot Probability", command=run_simulation_and_plot)
    plot_button.grid(row=0, column=4, padx=5, pady=5)

    simulate_replicas_button = ttk.Button(frame, text="Simulate Replicas", command=simulate_replicas)
    simulate_replicas_button.grid(row=0, column=5, padx=5, pady=5)
    simulate_replicas_button.config(state=tk.DISABLED)

    result_frame = ttk.Frame(frame)
    result_frame.grid(row=2, column=0, columnspan=6, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

    result_text = tk.Text(result_frame, wrap='none', height=20, width=100)
    result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    scrollbar_y = ttk.Scrollbar(result_frame, orient='vertical', command=result_text.yview)
    scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
    result_text['yscrollcommand'] = scrollbar_y.set

    scrollbar_x = ttk.Scrollbar(result_frame, orient='horizontal', command=result_text.xview)
    scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
    result_text['xscrollcommand'] = scrollbar_x.set

    app.mainloop()
