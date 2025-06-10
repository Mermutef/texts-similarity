def pretty_mean_report(df):
    metrics = df[['f1', 'precision', 'recall']]
    stats = metrics.agg(['mean', 'std']).transpose()
    
    # Используем моноширинное форматирование для Telegram
    table  = "┌────────────┬──────────┬──────────┐\n"
    table += "│   Метрика  │  Среднее │   СКО    │\n"
    table += "├────────────┼──────────┼──────────┤\n"
    
    for index, row in stats.iterrows():
        table += f"│ {index.ljust(10)} │  {row['mean']:.4f}  │  {row['std']:.4f}  │\n"
    
    table += "└────────────┴──────────┴──────────┘"
    
    return table