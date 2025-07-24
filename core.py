import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def create_experiment(df_clientes, df_kpis, cliente_id, kpi_principal, periodo_analisis_meses=6, num_seeds=10, seed_start=7190):
    """
    Crea grupos acción/control con métricas robustas de tendencias paralelas
    
    Parameters:
    -----------
    df_clientes : DataFrame
        DataFrame con información de clientes
    df_kpis : DataFrame  
        DataFrame con KPIs históricos por período
    cliente_id : str
        Nombre de la columna que identifica al cliente
    kpi_principal : str
        Nombre del KPI principal a optimizar
    periodo_analisis_meses : int
        Número de meses históricos a analizar (default: 6)
    num_seeds : int
        Número de semillas a probar para optimización (default: 10)
    seed_start : int
        Semilla inicial para comenzar la búsqueda (default: 7190)
    
    Returns:
    --------
    SimpleResult
        Objeto con grupos, quality_score, validation_report y plot
    """
    
    print(" INICIANDO DISEÑO EXPERIMENTAL AUTOMÁTICO")
    print("=" * 50)
    
    # 1. Preparar datos con KPIs agregados
    print(" Preparando datos...")
    df_preparado, ultimos_meses = _prepare_data(df_clientes, df_kpis, cliente_id, kpi_principal, periodo_analisis_meses)
    
    # 2. Optimización de semilla
    print(" Optimizando semilla...")
    best_seed = _optimize_seed(df_preparado, df_kpis, ultimos_meses, kpi_principal, cliente_id, num_seeds, seed_start)
    
    # 3. Crear grupos con mejor semilla
    print(" Creando grupos balanceados...")
    df_grupos = _create_balanced_groups(df_preparado, kpi_principal, best_seed)
    
    # 4. Validación completa
    print(" Validando calidad del diseño...")
    quality_metrics = _calculate_robust_metrics(df_grupos, df_kpis, ultimos_meses, kpi_principal, cliente_id)
    
    # 5. Generar gráfico de tendencias paralelas
    print(" Generando gráfico de tendencias...")
    parallel_trends_plot = _generate_parallel_trends_plot(df_grupos, df_kpis, ultimos_meses, kpi_principal, cliente_id)
    
    # 6. Reporte de validación
    validation_report = _generate_validation_report(df_grupos, quality_metrics, kpi_principal)
    
    print("✅ DISEÑO EXPERIMENTAL COMPLETADO")
    print(f"Quality Score: {quality_metrics['score_total']}/100")
    
    return SimpleResult(df_grupos, quality_metrics['score_total'], validation_report, parallel_trends_plot)

def _prepare_data(df_clientes, df_kpis, cliente_id, kpi_principal, meses):
    """Prepara datos agregando KPIs por cliente"""
    
    # Convertir periodo a string si es datetime
    if df_kpis['periodo'].dtype == 'datetime64[ns]' or df_kpis['periodo'].dtype == 'object':
        try:
            df_kpis = df_kpis.copy()
            df_kpis['periodo'] = pd.to_datetime(df_kpis['periodo']).dt.strftime('%Y-%m')
        except:
            pass
    
    # Últimos N meses (excluyendo mes actual)
    periodos = sorted(df_kpis['periodo'].unique())
    ultimos_meses = periodos[-(meses+1):-1]
    
    print(f"   - Analizando últimos {meses} meses: {ultimos_meses[0]} a {ultimos_meses[-1]}")
    print(f"   - Mes actual excluido: {periodos[-1]}")
    
    # KPIs por cliente (total, promedio)
    kpi_cliente = df_kpis[
        df_kpis['periodo'].isin(ultimos_meses)
    ].groupby(cliente_id).agg({
        kpi_principal: ['sum', 'mean', 'count']
    }).reset_index()
    
    # Aplanar columnas
    kpi_cliente.columns = [cliente_id, f'{kpi_principal}_total', f'{kpi_principal}_promedio', f'{kpi_principal}_periodos']
    kpi_cliente = kpi_cliente.fillna(0)
    
    # Filtrar clientes con datos suficientes
    kpi_cliente = kpi_cliente[kpi_cliente[f'{kpi_principal}_periodos'] >= 3]
    
    # Merge con clientes
    df_preparado = df_clientes.merge(kpi_cliente, on=cliente_id, how='inner')
    
    print(f"   - Clientes con datos suficientes: {len(df_preparado):,}")
    
    return df_preparado, ultimos_meses

def _optimize_seed(df_preparado, df_kpis, ultimos_meses, kpi_principal, cliente_id, num_seeds=10, seed_start=7190):
    """Optimización de semilla usando métricas robustas"""
    
    best_score = 0
    best_seed = seed_start
    
    print(f"    Probando {num_seeds} semillas desde {seed_start}...")
    
    for i, seed in enumerate(range(seed_start, seed_start + num_seeds)):
        # Crear grupos candidatos
        df_test = _create_balanced_groups(df_preparado, kpi_principal, seed)
        
        # Calcular métricas robustas
        metrics = _calculate_robust_metrics(df_test, df_kpis, ultimos_meses, kpi_principal, cliente_id)
        score = metrics['score_total']
        
        if score > best_score:
            best_score = score
            best_seed = seed
            if i % 3 == 0 or score > best_score:
                print(f"      Semilla {seed}: {score:.1f} ")
        elif i % 5 == 0:
            print(f"      Semilla {seed}: {score:.1f}")
    
    print(f"   ✅ Mejor semilla: {best_seed} (score: {best_score:.1f})")
    return best_seed

def _calculate_robust_metrics(df_grupos, df_kpis, ultimos_meses, kpi_principal, cliente_id):
    """Métricas robustas: Tendencias paralelas (70%) + Balance estadístico (30%)"""
    
    # Filtrar KPIs para clientes del experimento
    clientes_experimento = df_grupos[cliente_id].unique()
    df_kpis_exp = df_kpis[
        df_kpis[cliente_id].isin(clientes_experimento) &
        df_kpis['periodo'].isin(ultimos_meses)
    ].copy()
    
    # Convertir período a formato consistente
    df_kpis_exp['periodo_str'] = pd.to_datetime(df_kpis_exp['periodo']).dt.strftime('%Y-%m')
    
    # Merge con grupos
    df_evolucion = df_kpis_exp.merge(
        df_grupos[[cliente_id, 'grupo']], 
        on=cliente_id, 
        how='inner'
    )
    
    if len(df_evolucion) == 0:
        return {
            'score_total': 30.0,
            'score_paralelas': 0.0,
            'score_balance': 100.0,
            'peso_paralelas': 0.7,
            'peso_balance': 0.3,
            'interpretacion': "🔴 ERROR: Sin datos para análisis"
        }
    
    # 1. TENDENCIAS PARALELAS (70% peso)
    score_paralelas = _calculate_parallel_trends_score(df_evolucion, kpi_principal)
    
    # 2. BALANCE ESTADÍSTICO (30% peso)
    score_balance = _calculate_statistical_balance_score(df_grupos, kpi_principal)
    
    # 3. Score combinado
    peso_paralelas = 0.7
    peso_balance = 0.3
    score_total = score_paralelas * peso_paralelas + score_balance * peso_balance
    
    return {
        'score_total': round(score_total, 1),
        'score_paralelas': round(score_paralelas, 1),
        'score_balance': round(score_balance, 1),
        'peso_paralelas': peso_paralelas,
        'peso_balance': peso_balance,
        'interpretacion': _interpret_scores(score_total, score_paralelas, score_balance)
    }

def _calculate_parallel_trends_score(df_evolucion, kpi_principal):
    """Score de tendencias paralelas usando CV de diferencias con escala realista"""
    
    try:
        # Convertir KPI a numérico
        df_evolucion[kpi_principal] = pd.to_numeric(df_evolucion[kpi_principal], errors='coerce').fillna(0)
        
        # Verificar grupos disponibles
        grupos_disponibles = df_evolucion['grupo'].unique()
        if 'accion' not in grupos_disponibles or 'control' not in grupos_disponibles:
            return 50
        
        # Evolución promedio por período y grupo
        evol_summary = df_evolucion.groupby(['periodo_str', 'grupo'])[kpi_principal].mean().reset_index()
        
        if len(evol_summary) < 4:  # Al menos 2 períodos x 2 grupos
            return 50
        
        # Pivot para calcular diferencias
        pivot = evol_summary.pivot(index='periodo_str', columns='grupo', values=kpi_principal)
        
        if 'accion' not in pivot.columns or 'control' not in pivot.columns:
            return 50
        
        # Limpiar NaN y calcular diferencias
        pivot_clean = pivot.dropna()
        if len(pivot_clean) < 2:
            return 50
        
        pivot_clean['diferencia'] = pivot_clean['accion'] - pivot_clean['control']
        diferencias_clean = pivot_clean['diferencia'].dropna()
        
        if len(diferencias_clean) == 0:
            return 50
        
        # Calcular CV
        diferencia_mean = diferencias_clean.mean()
        diferencia_std = diferencias_clean.std()
        
        if abs(diferencia_mean) < 0.01:
            cv_diferencias = diferencia_std / max(pivot_clean['accion'].mean(), pivot_clean['control'].mean(), 1)
        else:
            cv_diferencias = abs(diferencia_std / diferencia_mean)
        
        # Convertir CV a score con escala realista
        if cv_diferencias <= 0.2:
            score_paralelas = 90 + (0.2 - cv_diferencias) * 50  # 90-100
        elif cv_diferencias <= 0.5:
            score_paralelas = 70 + (0.5 - cv_diferencias) * 66.67  # 70-90
        elif cv_diferencias <= 1.0:
            score_paralelas = 50 + (1.0 - cv_diferencias) * 40  # 50-70
        elif cv_diferencias <= 2.0:
            score_paralelas = 20 + (2.0 - cv_diferencias) * 30  # 20-50
        else:
            score_paralelas = max(0, 20 - (cv_diferencias - 2.0) * 10)  # 0-20
        
        return max(0, min(100, score_paralelas))
        
    except Exception:
        return 50

def _calculate_statistical_balance_score(df_grupos, kpi_principal):
    """Score de balance estadístico usando t-test"""
    
    try:
        col_kpi_total = f'{kpi_principal}_total'
        
        if col_kpi_total not in df_grupos.columns:
            return 50
        
        accion = df_grupos[df_grupos['grupo'] == 'accion'][col_kpi_total]
        control = df_grupos[df_grupos['grupo'] == 'control'][col_kpi_total]
        
        if len(accion) == 0 or len(control) == 0:
            return 50
        
        # t-test para balance
        _, p_value = stats.ttest_ind(accion, control)
        
        # Convertir p-value a score
        score_balance = min(p_value * 100, 100)
        
        return score_balance
        
    except Exception:
        return 50

def _create_balanced_groups(df, kpi_principal, seed=7190):
    """Crear grupos balanceados por quintiles"""
    
    np.random.seed(seed)
    df = df.copy()
    
    # Usar KPI total para estratificación
    col_kpi = f'{kpi_principal}_total'
    if col_kpi not in df.columns:
        col_kpi = f'{kpi_principal}_promedio'
    
    # Crear quintiles por KPI
    try:
        df['kpi_quintil'] = pd.qcut(df[col_kpi], q=5, labels=False, duplicates='drop') + 1
    except:
        df['kpi_quintil'] = 1
    
    # Inicializar columna grupo
    df['grupo'] = 'control'
    
    # Balancear por quintil
    for quintil in df['kpi_quintil'].unique():
        subset_idx = df[df['kpi_quintil'] == quintil].index
        subset_shuffled = np.random.permutation(subset_idx)
        
        # 50/50 split
        mitad = len(subset_shuffled) // 2
        accion_idx = subset_shuffled[:mitad]
        
        df.loc[accion_idx, 'grupo'] = 'accion'
    
    return df

def _generate_parallel_trends_plot(df_grupos, df_kpis, ultimos_meses, kpi_principal, cliente_id):
    """Genera gráfico de tendencias paralelas mejorado"""
    
    # Filtrar KPIs para clientes del experimento
    df_kpis_exp = df_kpis[
        df_kpis[cliente_id].isin(df_grupos[cliente_id]) &
        df_kpis['periodo'].isin(ultimos_meses)
    ].copy()
    
    # Convertir período y merge con grupos
    df_kpis_exp['periodo_str'] = pd.to_datetime(df_kpis_exp['periodo']).dt.strftime('%Y-%m')
    df_evolucion = df_kpis_exp.merge(
        df_grupos[[cliente_id, 'grupo']], 
        on=cliente_id, 
        how='inner'
    )
    
    # Convertir KPI a numérico
    df_evolucion[kpi_principal] = pd.to_numeric(df_evolucion[kpi_principal], errors='coerce').fillna(0)
    
    # Evolución promedio por período y grupo
    evol_summary = df_evolucion.groupby(['periodo_str', 'grupo'])[kpi_principal].mean().reset_index()
    
    # Crear el gráfico
    plt.figure(figsize=(14, 8))
    
    # Colores modernos
    colors = {'accion': '#2E86AB', 'control': '#A23B72'}
    
    for grupo in ['accion', 'control']:
        data_grupo = evol_summary[evol_summary['grupo'] == grupo]
        if len(data_grupo) > 0:
            plt.plot(data_grupo['periodo_str'], data_grupo[kpi_principal], 
                    marker='o', linewidth=3, markersize=8,
                    label=f'Grupo {grupo.title()}',
                    color=colors[grupo])
    
    # Personalización del gráfico
    plt.title(f'Validación de Tendencias Paralelas - {kpi_principal}\n(Análisis Pre-Experimento)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Período', fontsize=12, fontweight='bold')
    plt.ylabel(f'{kpi_principal} Promedio', fontsize=12, fontweight='bold')
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Formato de ejes
    plt.xticks(rotation=45)
    
    # Calcular y mostrar métricas de paralelismo
    _add_parallelism_metrics_to_plot(evol_summary, kpi_principal)
    
    plt.tight_layout()
    
    return plt.gcf()

def _add_parallelism_metrics_to_plot(evol_summary, kpi_principal):
    """Agrega métricas de paralelismo al gráfico con escala corregida"""
    
    try:
        # Verificar que tenemos datos válidos
        if len(evol_summary) == 0:
            plt.figtext(0.02, 0.02, 'No hay datos para calcular métricas', fontsize=10, color='red')
            return
        
        # Pivot para calcular diferencias
        pivot = evol_summary.pivot(index='periodo_str', columns='grupo', values=kpi_principal)
        
        if 'accion' not in pivot.columns or 'control' not in pivot.columns:
            plt.figtext(0.02, 0.02, 'Faltan grupos para calcular paralelismo', fontsize=10, color='red')
            return
        
        # Limpiar y calcular diferencias
        pivot_clean = pivot.dropna()
        if len(pivot_clean) < 2:
            plt.figtext(0.02, 0.02, 'Muy pocos períodos para análisis', fontsize=10, color='orange')
            return
        
        pivot_clean['diferencia'] = pivot_clean['accion'] - pivot_clean['control']
        diferencias_clean = pivot_clean['diferencia'].dropna()
        
        if len(diferencias_clean) == 0:
            plt.figtext(0.02, 0.02, 'No se pueden calcular diferencias válidas', fontsize=10, color='red')
            return
        
        # CV de diferencias
        diferencia_mean = diferencias_clean.mean()
        diferencia_std = diferencias_clean.std()
        
        if abs(diferencia_mean) < 0.01:
            cv_diferencias = diferencia_std / max(pivot_clean['accion'].mean(), pivot_clean['control'].mean(), 1)
        else:
            cv_diferencias = abs(diferencia_std / diferencia_mean)
        
        # Interpretación con nueva escala
        if cv_diferencias <= 0.2:
            status = "✅ EXCELENTE"
            color = 'green'
        elif cv_diferencias <= 0.5:
            status = "🟡 BUENO" 
            color = 'orange'
        elif cv_diferencias <= 1.0:
            status = "🟠 ACEPTABLE"
            color = 'darkorange'
        elif cv_diferencias <= 2.0:
            status = "🔴 REGULAR"
            color = 'red'
        else:
            status = "❌ MALO"
            color = 'darkred'
        
        # Agregar texto al gráfico
        plt.figtext(0.02, 0.02, 
                   f'CV Diferencias: {cv_diferencias:.3f} | Evaluación Paralelismo: {status}',
                   fontsize=11, fontweight='bold', color=color,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
    except Exception as e:
        plt.figtext(0.02, 0.02, 
                   f'Error calculando métricas: {str(e)[:50]}...',
                   fontsize=10, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

def _generate_validation_report(df_grupos, metrics, kpi_principal):
    """Reporte de validación con métricas robustas"""
    
    accion = df_grupos[df_grupos['grupo'] == 'accion']
    control = df_grupos[df_grupos['grupo'] == 'control']
    
    # KPI promedio por grupo
    col_kpi = f'{kpi_principal}_total'
    if col_kpi in df_grupos.columns:
        kpi_accion = accion[col_kpi].mean()
        kpi_control = control[col_kpi].mean()
        diferencia_abs = abs(kpi_accion - kpi_control)
        diferencia_rel = (diferencia_abs / max(kpi_accion, kpi_control) * 100) if max(kpi_accion, kpi_control) > 0 else 0
    else:
        kpi_accion = kpi_control = diferencia_abs = diferencia_rel = 0
    
    report = f"""
VALIDACIÓN EXPERIMENTAL ROBUSTA
==============================
 BALANCE DE GRUPOS:
   • Total clientes: {len(df_grupos):,}
   • Grupo acción: {len(accion):,} ({len(accion)/len(df_grupos)*100:.1f}%)
   • Grupo control: {len(control):,} ({len(control)/len(df_grupos)*100:.1f}%)

 MÉTRICAS DE CALIDAD:
   • Score Total: {metrics['score_total']}/100
   • Tendencias Paralelas: {metrics['score_paralelas']}/100 (peso: {metrics['peso_paralelas']*100:.0f}%)
   • Balance Estadístico: {metrics['score_balance']}/100 (peso: {metrics['peso_balance']*100:.0f}%)

 KPI PRINCIPAL ({kpi_principal}):
   • Promedio Acción: {kpi_accion:.2f}
   • Promedio Control: {kpi_control:.2f}
   • Diferencia Absoluta: {diferencia_abs:.2f}
   • Diferencia Relativa: {diferencia_rel:.1f}%

 RECOMENDACIÓN:
   {metrics['interpretacion']}

 PRÓXIMOS PASOS:
   1. Revisar gráfico de tendencias paralelas
   2. Exportar grupos para implementación
   3. Establecer métricas de seguimiento
"""
    return report.strip()

def _interpret_scores(score_total, score_paralelas, score_balance):
    """Interpretación de scores"""
    if score_total >= 80:
        return "🟢 EXCELENTE - Diseño robusto, proceder con experimento"
    elif score_total >= 65:
        return "🟡 BUENO - Diseño aceptable, considerar validaciones adicionales"
    elif score_total >= 50:
        return "🟠 REGULAR - Revisar datos o considerar más período de análisis"
    else:
        return "🔴 MEJORABLE - Recomendado revisar calidad de datos o estrategia"

class SimpleResult:
    """Resultado con métricas robustas y gráfico de tendencias"""
    def __init__(self, grupos, quality_score, validation_report, parallel_trends_plot):
        self.grupos = grupos
        self.quality_score = quality_score
        self.validation_report = validation_report
        self.parallel_trends_plot = parallel_trends_plot
    
    def show_plot(self):
        """Muestra el gráfico de tendencias paralelas"""
        self.parallel_trends_plot.show()
    
    def save_results(self, prefix="experimento"):
        """Guarda todos los resultados a archivos"""
        
        # Guardar DataFrame de grupos
        self.grupos.to_csv(f"{prefix}_grupos.csv", index=False)
        
        # Guardar reporte
        with open(f"{prefix}_reporte.txt", "w", encoding='utf-8') as f:
            f.write(self.validation_report)
        
        # Guardar gráfico
        self.parallel_trends_plot.savefig(f"{prefix}_tendencias_paralelas.png", 
                                         dpi=300, bbox_inches='tight')
        
        print(f"✅ Resultados guardados:")
        print(f"   - {prefix}_grupos.csv")
        print(f"   - {prefix}_reporte.txt") 
        print(f"   - {prefix}_tendencias_paralelas.png")
    
    def summary(self):
        """Resumen ejecutivo"""
        accion = len(self.grupos[self.grupos['grupo'] == 'accion'])
        control = len(self.grupos[self.grupos['grupo'] == 'control'])
        
        return {
            'total_clientes': len(self.grupos),
            'grupo_accion': accion,
            'grupo_control': control,
            'balance_ratio': round(accion / control, 3),
            'quality_score': self.quality_score
        }


