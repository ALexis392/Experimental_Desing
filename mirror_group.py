import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def find_mirror_group(df_clientes, df_kpis, grupo_definido, cliente_id, kpi_principal, 
                     periodo_analisis_meses=6, num_seeds=20, seed_start=7190, 
                     grupo_definido_tipo='accion'):
    """
    Encuentra el grupo espejo (control/acción) dado un grupo ya definido
    
    Parameters:
    -----------
    df_clientes : DataFrame
        DataFrame con información de todos los clientes candidatos
    df_kpis : DataFrame  
        DataFrame con KPIs históricos por período
    grupo_definido : DataFrame o list
        DataFrame con clientes del grupo ya definido, o lista de IDs
    cliente_id : str
        Nombre de la columna que identifica al cliente
    kpi_principal : str
        Nombre del KPI principal a optimizar
    periodo_analisis_meses : int
        Número de meses históricos a analizar (default: 6)
    num_seeds : int
        Número de semillas a probar para optimización (default: 20)
    seed_start : int
        Semilla inicial para comenzar la búsqueda (default: 7190)
    grupo_definido_tipo : str
        Tipo del grupo definido: 'accion' o 'control' (default: 'accion')
    
    Returns:
    --------
    MirrorResult
        Objeto con ambos grupos, quality_score, validation_report y plot
    """
    
    print("🔍 BÚSQUEDA DE GRUPO ESPEJO CON OPTIMIZACIÓN ROBUSTA")
    print("=" * 50)
    
    # 1. Preparar grupo definido
    if isinstance(grupo_definido, list):
        ids_grupo_definido = grupo_definido
        print(f" Grupo {grupo_definido_tipo} definido: {len(ids_grupo_definido)} clientes (lista de IDs)")
    else:
        ids_grupo_definido = grupo_definido[cliente_id].tolist()
        print(f" Grupo {grupo_definido_tipo} definido: {len(ids_grupo_definido)} clientes (DataFrame)")
    
    # 2. Preparar datos base
    print(" Preparando datos...")
    df_preparado, ultimos_meses = _prepare_data_mirror(df_clientes, df_kpis, cliente_id, kpi_principal, periodo_analisis_meses)
    
    # 3. Crear conjunto de candidatos (excluir grupo definido)
    candidatos = df_preparado[~df_preparado[cliente_id].isin(ids_grupo_definido)].copy()
    print(f"   - Candidatos disponibles: {len(candidatos):,}")
    
    # 4. Analizar características del grupo definido
    grupo_referencia = df_preparado[df_preparado[cliente_id].isin(ids_grupo_definido)].copy()
    estadisticas_ref = _analyze_reference_group(grupo_referencia, kpi_principal)
    print(f"   - Estadísticas grupo referencia calculadas")
    
    # 5. OPTIMIZACIÓN ROBUSTA para encontrar mejor espejo (similar a core.py)
    print(" Optimizando grupo espejo con métricas robustas...")
    best_mirror, best_score, best_seed = _optimize_mirror_robust(
        grupo_referencia, candidatos, df_kpis, ultimos_meses, 
        kpi_principal, cliente_id, estadisticas_ref, num_seeds, seed_start
    )
    
    # 6. Crear DataFrame final con ambos grupos
    df_final = _create_final_groups(grupo_referencia, best_mirror, grupo_definido_tipo)
    
    # 7. Validación completa usando métricas robustas
    print(" Validando calidad del espejo...")
    quality_metrics = _calculate_mirror_quality_robust(df_final, df_kpis, ultimos_meses, kpi_principal, cliente_id)
    
    # 8. Generar visualizaciones
    print(" Generando visualizaciones...")
    comparison_plot = _generate_mirror_comparison_plot(df_final, df_kpis, ultimos_meses, kpi_principal, cliente_id)
    
    # 9. Reporte de validación
    validation_report = _generate_mirror_report(df_final, quality_metrics, kpi_principal, grupo_definido_tipo, best_seed)
    
    print(" GRUPO ESPEJO ENCONTRADO")
    print(f"Quality Score: {quality_metrics['score_total']}/100")
    
    return MirrorResult(df_final, quality_metrics['score_total'], validation_report, comparison_plot, estadisticas_ref)

def _optimize_mirror_robust(grupo_ref, candidatos, df_kpis, ultimos_meses, kpi_principal, 
                           cliente_id, stats_ref, num_seeds=20, seed_start=7190):
    """
    Optimización robusta de semilla para grupo espejo (similar a core.py)
    Usa métricas robustas: Tendencias paralelas (70%) + Balance estadístico (30%)
    """
    
    target_size = len(grupo_ref)
    
    best_score = 0
    best_mirror = None
    best_seed = seed_start
    
    print(f"   🔍 Probando {num_seeds} semillas desde {seed_start}...")
    
    for i, seed in enumerate(range(seed_start, seed_start + num_seeds)):
        
        if (i + 1) % 5 == 0:
            print(f"      Evaluadas {i + 1}/{num_seeds} semillas...")
        
        try:
            # Crear espejo candidato con esta semilla
            mirror_candidate = _create_stratified_mirror_robust(
                candidatos, stats_ref, target_size, kpi_principal, seed
            )
            
            if len(mirror_candidate) == 0:
                continue
            
            # Crear grupos temporales para evaluación
            df_temp = _create_final_groups(grupo_ref, mirror_candidate, 'accion')
            
            # USAR MÉTRICAS ROBUSTAS (similar a core.py)
            metrics = _calculate_mirror_quality_robust(df_temp, df_kpis, ultimos_meses, kpi_principal, cliente_id)
            score = metrics['score_total']
            
            if score > best_score:
                best_score = score
                best_mirror = mirror_candidate.copy()
                best_seed = seed
                if i % 3 == 0 or score > best_score:
                    print(f"         Semilla {seed}: {score:.1f} ")
            elif i % 5 == 0:
                print(f"         Semilla {seed}: {score:.1f}")
                
        except Exception as e:
            print(f"         Error con semilla {seed}: {str(e)[:50]}...")
            continue
    
    print(f"   ✅ Mejor espejo: semilla {best_seed} (score: {best_score:.1f})")
    
    return best_mirror, best_score, best_seed

def _create_stratified_mirror_robust(candidatos, stats_ref, target_size, kpi_principal, seed):
    """
    Crea grupo espejo usando estratificación robusta (mejorado)
    """
    
    col_kpi_total = f'{kpi_principal}_total'
    np.random.seed(seed)
    
    if len(candidatos) < target_size:
        return candidatos.sample(n=len(candidatos), random_state=seed)
    
    # Estrategia mejorada: Quintiles + matching por rango
    try:
        # 1. Crear quintiles en candidatos
        candidatos_copy = candidatos.copy()
        candidatos_copy['kpi_quintil'] = pd.qcut(
            candidatos_copy[col_kpi_total], 
            q=5, 
            labels=False, 
            duplicates='drop'
        ) + 1
        
        # 2. Si tenemos distribución de referencia, usarla para estratificar
        if stats_ref.get('decil_distribution'):
            # Convertir deciles a quintiles
            quintil_distribution = {}
            for decil, prop in stats_ref['decil_distribution'].items():
                quintil = min(5, (decil + 1) // 2)  # Mapear deciles a quintiles
                quintil_distribution[quintil] = quintil_distribution.get(quintil, 0) + prop
        else:
            # Distribución uniforme por quintiles
            quintil_distribution = {i: 0.2 for i in range(1, 6)}
        
        # 3. Seleccionar por quintiles respetando proporciones
        mirror_parts = []
        for quintil, proporcion in quintil_distribution.items():
            n_needed = int(target_size * proporcion)
            if n_needed == 0:
                continue
                
            quintil_candidates = candidatos_copy[candidatos_copy['kpi_quintil'] == quintil]
            
            if len(quintil_candidates) >= n_needed:
                # Weighted sampling dentro del quintil (priorizar similitud)
                target_mean = stats_ref['kpi_mean']
                quintil_candidates['similarity_weight'] = 1 / (1 + abs(quintil_candidates[col_kpi_total] - target_mean))
                weights = quintil_candidates['similarity_weight'] / quintil_candidates['similarity_weight'].sum()
                
                selected_indices = np.random.choice(
                    quintil_candidates.index, 
                    size=n_needed, 
                    replace=False, 
                    p=weights
                )
                selected = candidatos_copy.loc[selected_indices]
            else:
                selected = quintil_candidates
            
            mirror_parts.append(selected)
        
        if mirror_parts:
            mirror_grupo = pd.concat(mirror_parts, ignore_index=True)
            
            # Si no llegamos al tamaño target, completar con mejores candidatos restantes
            if len(mirror_grupo) < target_size:
                remaining_candidates = candidatos_copy[~candidatos_copy.index.isin(mirror_grupo.index)]
                if len(remaining_candidates) > 0:
                    n_additional = min(target_size - len(mirror_grupo), len(remaining_candidates))
                    
                    # Priorizar por similitud de KPI
                    target_mean = stats_ref['kpi_mean']
                    remaining_candidates['similarity'] = abs(remaining_candidates[col_kpi_total] - target_mean)
                    additional = remaining_candidates.nsmallest(n_additional, 'similarity')
                    mirror_grupo = pd.concat([mirror_grupo, additional], ignore_index=True)
            
            return mirror_grupo.head(target_size)
    
    except Exception:
        # Fallback: selección aleatoria weighted
        pass
    
    # Fallback robusto
    target_mean = stats_ref['kpi_mean']
    candidatos['similarity_weight'] = 1 / (1 + abs(candidatos[col_kpi_total] - target_mean))
    weights = candidatos['similarity_weight'] / candidatos['similarity_weight'].sum()
    
    selected_indices = np.random.choice(
        candidatos.index, 
        size=min(target_size, len(candidatos)), 
        replace=False, 
        p=weights
    )
    
    return candidatos.loc[selected_indices]

def _calculate_mirror_quality_robust(df_grupos, df_kpis, ultimos_meses, kpi_principal, cliente_id):
    """
    Calcula métricas robustas para grupo espejo (IGUAL que core.py)
    Tendencias paralelas (70%) + Balance estadístico (30%)
    """
    
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
    
    # 1. TENDENCIAS PARALELAS (70% peso) - IGUAL que core.py
    score_paralelas = _calculate_parallel_trends_score_robust(df_evolucion, kpi_principal)
    
    # 2. BALANCE ESTADÍSTICO (30% peso) - IGUAL que core.py  
    score_balance = _calculate_statistical_balance_score_robust(df_grupos, kpi_principal)
    
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
        'interpretacion': _interpret_scores_robust(score_total, score_paralelas, score_balance)
    }

def _calculate_parallel_trends_score_robust(df_evolucion, kpi_principal):
    """
    Score de tendencias paralelas usando CV de diferencias (COPIADO de core.py)
    """
    
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
        
        # Convertir CV a score con escala realista (IGUAL que core.py)
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

def _calculate_statistical_balance_score_robust(df_grupos, kpi_principal):
    """
    Score de balance estadístico usando t-test (COPIADO de core.py)
    """
    
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

def _interpret_scores_robust(score_total, score_paralelas, score_balance):
    """Interpretación de scores (IGUAL que core.py)"""
    if score_total >= 80:
        return "🟢 EXCELENTE - Grupo espejo robusto, proceder con experimento"
    elif score_total >= 65:
        return "🟡 BUENO - Grupo espejo aceptable, considerar validaciones adicionales"
    elif score_total >= 50:
        return "🟠 REGULAR - Revisar datos o considerar más candidatos"
    else:
        return "🔴 MEJORABLE - Recomendado revisar calidad de datos o estrategia"

# ============================================================================
# FUNCIONES AUXILIARES (mantener las originales)
# ============================================================================

def _prepare_data_mirror(df_clientes, df_kpis, cliente_id, kpi_principal, meses):
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
    
    # KPIs por cliente
    kpi_cliente = df_kpis[
        df_kpis['periodo'].isin(ultimos_meses)
    ].groupby(cliente_id).agg({
        kpi_principal: ['sum', 'mean', 'std', 'count']
    }).reset_index()
    
    # Aplanar columnas
    kpi_cliente.columns = [cliente_id, f'{kpi_principal}_total', f'{kpi_principal}_promedio', 
                          f'{kpi_principal}_std', f'{kpi_principal}_periodos']
    kpi_cliente = kpi_cliente.fillna(0)
    
    # Filtrar clientes con datos suficientes
    kpi_cliente = kpi_cliente[kpi_cliente[f'{kpi_principal}_periodos'] >= 3]
    
    # Merge con clientes
    df_preparado = df_clientes.merge(kpi_cliente, on=cliente_id, how='inner')
    
    print(f"   - Clientes con datos suficientes: {len(df_preparado):,}")
    
    return df_preparado, ultimos_meses

def _analyze_reference_group(grupo_ref, kpi_principal):
    """Analiza características del grupo de referencia"""
    
    col_kpi_total = f'{kpi_principal}_total'
    col_kpi_promedio = f'{kpi_principal}_promedio'
    col_kpi_std = f'{kpi_principal}_std'
    
    stats_ref = {
        'size': len(grupo_ref),
        'kpi_mean': grupo_ref[col_kpi_total].mean(),
        'kpi_std': grupo_ref[col_kpi_total].std(),
        'kpi_median': grupo_ref[col_kpi_total].median(),
        'kpi_promedio_mean': grupo_ref[col_kpi_promedio].mean(),
        'kpi_variabilidad_mean': grupo_ref[col_kpi_std].mean()
    }
    
    # Calcular deciles para estratificación
    if len(grupo_ref) >= 10:
        try:
            grupo_ref_copy = grupo_ref.copy()
            grupo_ref_copy['kpi_decil'] = pd.qcut(grupo_ref_copy[col_kpi_total], q=10, labels=False, duplicates='drop') + 1
            decil_distribution = grupo_ref_copy['kpi_decil'].value_counts(normalize=True).sort_index()
            stats_ref['decil_distribution'] = decil_distribution.to_dict()
        except:
            stats_ref['decil_distribution'] = {}
    else:
        stats_ref['decil_distribution'] = {}
    
    return stats_ref

def _create_final_groups(grupo_ref, grupo_mirror, tipo_ref):
    """Crea DataFrame final con ambos grupos etiquetados"""
    
    grupo_ref_copy = grupo_ref.copy()
    grupo_mirror_copy = grupo_mirror.copy()
    
    grupo_ref_copy['grupo'] = tipo_ref
    grupo_mirror_copy['grupo'] = 'control' if tipo_ref == 'accion' else 'accion'
    
    df_final = pd.concat([grupo_ref_copy, grupo_mirror_copy], ignore_index=True)
    
    return df_final

def _generate_mirror_comparison_plot(df_grupos, df_kpis, ultimos_meses, kpi_principal, cliente_id):
    """Genera gráfico de tendencias temporales únicamente (simplificado)"""
    
    # Filtrar y preparar datos
    df_kpis_exp = df_kpis[
        df_kpis[cliente_id].isin(df_grupos[cliente_id]) &
        df_kpis['periodo'].isin(ultimos_meses)
    ].copy()
    
    df_kpis_exp['periodo_str'] = pd.to_datetime(df_kpis_exp['periodo']).dt.strftime('%Y-%m')
    df_evolucion = df_kpis_exp.merge(df_grupos[[cliente_id, 'grupo']], on=cliente_id, how='inner')
    df_evolucion[kpi_principal] = pd.to_numeric(df_evolucion[kpi_principal], errors='coerce').fillna(0)
    
    # Evolución promedio por período y grupo
    evol_summary = df_evolucion.groupby(['periodo_str', 'grupo'])[kpi_principal].mean().reset_index()
    
    # Crear figura simple con un solo gráfico
    plt.figure(figsize=(12, 8))
    
    # Colores modernos
    colors = {'accion': '#2E86AB', 'control': '#A23B72'}
    
    # Graficar líneas de tendencia
    for grupo in ['accion', 'control']:
        data_grupo = evol_summary[evol_summary['grupo'] == grupo]
        if len(data_grupo) > 0:
            plt.plot(data_grupo['periodo_str'], data_grupo[kpi_principal], 
                    marker='o', linewidth=4, markersize=10,
                    label=f'Grupo {grupo.title()}', 
                    color=colors[grupo],
                    markerfacecolor=colors[grupo],
                    markeredgecolor='white',
                    markeredgewidth=2)
    
    # Personalización del gráfico
    plt.title(f'Validación de Tendencias Paralelas - {kpi_principal}\n(Análisis Pre-Experimento)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Período', fontsize=14, fontweight='bold')
    plt.ylabel(f'{kpi_principal} Promedio', fontsize=14, fontweight='bold')
    plt.legend(fontsize=13, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Mejorar formato de ejes
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Añadir métricas de paralelismo en el gráfico
    _add_parallelism_metrics_to_plot_simple(evol_summary, kpi_principal)
    
    plt.tight_layout()
    
    return plt.gcf()

def _add_parallelism_metrics_to_plot_simple(evol_summary, kpi_principal):
    """Agrega métricas de paralelismo al gráfico simplificado"""
    
    try:
        # Verificar que tenemos datos válidos
        if len(evol_summary) == 0:
            plt.figtext(0.02, 0.02, 'No hay datos para calcular métricas', fontsize=12, color='red')
            return
        
        # Pivot para calcular diferencias
        pivot = evol_summary.pivot(index='periodo_str', columns='grupo', values=kpi_principal)
        
        if 'accion' not in pivot.columns or 'control' not in pivot.columns:
            plt.figtext(0.02, 0.02, 'Faltan grupos para calcular paralelismo', fontsize=12, color='red')
            return
        
        # Limpiar y calcular diferencias
        pivot_clean = pivot.dropna()
        if len(pivot_clean) < 2:
            plt.figtext(0.02, 0.02, 'Muy pocos períodos para análisis', fontsize=12, color='orange')
            return
        
        pivot_clean['diferencia'] = pivot_clean['accion'] - pivot_clean['control']
        diferencias_clean = pivot_clean['diferencia'].dropna()
        
        if len(diferencias_clean) == 0:
            plt.figtext(0.02, 0.02, 'No se pueden calcular diferencias válidas', fontsize=12, color='red')
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
        
        # Agregar texto al gráfico (posición mejorada)
        plt.figtext(0.02, 0.02, 
                   f'CV Diferencias: {cv_diferencias:.3f} | Evaluación Paralelismo: {status}',
                   fontsize=12, fontweight='bold', color=color,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, edgecolor=color))
        
    except Exception as e:
        plt.figtext(0.02, 0.02, 
                   f'Error calculando métricas: {str(e)[:50]}...',
                   fontsize=11, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))


def _generate_mirror_report(df_grupos, metrics, kpi_principal, tipo_ref, seed_used):
    """Genera reporte del análisis de grupo espejo"""
    
    grupo_ref = df_grupos[df_grupos['grupo'] == tipo_ref]
    grupo_espejo = df_grupos[df_grupos['grupo'] != tipo_ref]
    
    col_kpi = f'{kpi_principal}_total'
    if col_kpi in df_grupos.columns:
        kpi_ref = grupo_ref[col_kpi].mean()
        kpi_espejo = grupo_espejo[col_kpi].mean()
        diferencia_abs = abs(kpi_ref - kpi_espejo)
        diferencia_rel = (diferencia_abs / max(kpi_ref, kpi_espejo) * 100) if max(kpi_ref, kpi_espejo) > 0 else 0
    else:
        kpi_ref = kpi_espejo = diferencia_abs = diferencia_rel = 0
    
    report = f"""
ANÁLISIS DE GRUPO ESPEJO CON OPTIMIZACIÓN ROBUSTA
=================================================
 GRUPOS IDENTIFICADOS:
   • Grupo {tipo_ref.title()} (definido): {len(grupo_ref):,} clientes
   • Grupo Espejo (encontrado): {len(grupo_espejo):,} clientes
   • Semilla óptima utilizada: {seed_used}

 SIMILITUD DE GRUPOS:
   • {kpi_principal} promedio {tipo_ref}: {kpi_ref:.2f}
   • {kpi_principal} promedio espejo: {kpi_espejo:.2f}
   • Diferencia absoluta: {diferencia_abs:.2f}
   • Diferencia relativa: {diferencia_rel:.1f}%

 MÉTRICAS ROBUSTAS DE CALIDAD:
   • Score Total: {metrics['score_total']}/100
   • Tendencias Paralelas: {metrics['score_paralelas']}/100 (peso: {metrics['peso_paralelas']*100:.0f}%)
   • Balance Estadístico: {metrics['score_balance']}/100 (peso: {metrics['peso_balance']*100:.0f}%)

 EVALUACIÓN:
   {metrics['interpretacion']}

 RECOMENDACIONES:
   •  Revisar gráficos de similitud y tendencias paralelas
   •  Validar que ambos grupos son representativos del universo
   •  Considerar factores externos no medidos que puedan afectar
   •  Establecer métricas de seguimiento continuo del experimento
   •  Definir criterios de éxito y stopping rules
"""
    return report.strip()

class MirrorResult:
    """Resultado de búsqueda de grupo espejo con optimización robusta"""
    def __init__(self, grupos, quality_score, validation_report, comparison_plot, reference_stats):
        self.grupos = grupos
        self.quality_score = quality_score
        self.validation_report = validation_report
        self.comparison_plot = comparison_plot
        self.reference_stats = reference_stats
    
    def show_comparison(self):
        """Muestra gráfico de comparación"""
        self.comparison_plot.show()
    
    def save_results(self, prefix="grupo_espejo_robusto"):
        """Guarda todos los resultados"""
        
        # Guardar grupos
        self.grupos.to_csv(f"{prefix}_grupos_completos.csv", index=False)
        
        # Guardar cada grupo por separado
        grupo_ref = self.grupos[self.grupos['grupo'] == 'accion']
        grupo_espejo = self.grupos[self.grupos['grupo'] == 'control']
        
        grupo_ref.to_csv(f"{prefix}_grupo_accion.csv", index=False)
        grupo_espejo.to_csv(f"{prefix}_grupo_control.csv", index=False)
        
        # Guardar reporte
        with open(f"{prefix}_reporte.txt", "w", encoding='utf-8') as f:
            f.write(self.validation_report)
        
        # Guardar gráfico
        self.comparison_plot.savefig(f"{prefix}_comparacion.png", dpi=300, bbox_inches='tight')
        
        print(f" Resultados guardados:")
        print(f"   - {prefix}_grupos_completos.csv")
        print(f"   - {prefix}_grupo_accion.csv") 
        print(f"   - {prefix}_grupo_control.csv")
        print(f"   - {prefix}_reporte.txt")
        print(f"   - {prefix}_comparacion.png")
    
    def get_mirror_group(self, grupo_tipo='control'):
        """Obtiene solo el grupo espejo encontrado"""
        return self.grupos[self.grupos['grupo'] == grupo_tipo].copy()
    
    def summary(self):
        """Resumen ejecutivo"""
        grupo_accion = self.grupos[self.grupos['grupo'] == 'accion']
        grupo_control = self.grupos[self.grupos['grupo'] == 'control']
        
        return {
            'total_clientes': len(self.grupos),
            'grupo_accion': len(grupo_accion),
            'grupo_control': len(grupo_control),
            'similitud_score': self.quality_score,
            'estadisticas_referencia': self.reference_stats
        }

# ============================================================================
# FUNCIONES DE CONVENIENCIA PARA DIFERENTES CASOS DE USO
# ============================================================================

def find_control_for_action(df_clientes, df_kpis, grupo_accion, cliente_id, kpi_principal, 
                           periodo_analisis_meses=6, num_seeds=20):
    """
    Función de conveniencia: encuentra grupo control para un grupo acción ya definido
    """
    return find_mirror_group(
        df_clientes=df_clientes,
        df_kpis=df_kpis,
        grupo_definido=grupo_accion,
        cliente_id=cliente_id,
        kpi_principal=kpi_principal,
        periodo_analisis_meses=periodo_analisis_meses,
        num_seeds=num_seeds,
        grupo_definido_tipo='accion'
    )

def find_action_for_control(df_clientes, df_kpis, grupo_control, cliente_id, kpi_principal,
                           periodo_analisis_meses=6, num_seeds=20):
    """
    Función de conveniencia: encuentra grupo acción para un grupo control ya definido
    """
    return find_mirror_group(
        df_clientes=df_clientes,
        df_kpis=df_kpis,
        grupo_definido=grupo_control,
        cliente_id=cliente_id,
        kpi_principal=kpi_principal,
        periodo_analisis_meses=periodo_analisis_meses,
        num_seeds=num_seeds,
        grupo_definido_tipo='control'
    )

def optimize_mirror_seeds_batch(df_clientes, df_kpis, grupo_definido, cliente_id, kpi_principal,
                               seed_ranges=[(7190, 20), (8000, 20), (9000, 20)],
                               periodo_analisis_meses=6):
    """
    Optimización en lotes de diferentes rangos de semillas para encontrar el mejor espejo
    """
    print("🔍 OPTIMIZACIÓN BATCH DE SEMILLAS PARA GRUPO ESPEJO")
    print("=" * 55)
    
    best_result = None
    best_score = 0
    all_results = []
    
    for i, (seed_start, num_seeds) in enumerate(seed_ranges):
        print(f"\n📊 Lote {i+1}: Semillas {seed_start}-{seed_start+num_seeds-1}")
        
        result = find_mirror_group(
            df_clientes=df_clientes,
            df_kpis=df_kpis,
            grupo_definido=grupo_definido,
            cliente_id=cliente_id,
            kpi_principal=kpi_principal,
            periodo_analisis_meses=periodo_analisis_meses,
            num_seeds=num_seeds,
            seed_start=seed_start
        )
        
        all_results.append({
            'seed_range': f"{seed_start}-{seed_start+num_seeds-1}",
            'score': result.quality_score,
            'result': result
        })
        
        if result.quality_score > best_score:
            best_score = result.quality_score
            best_result = result
            print(f"   🌟 Nuevo mejor score: {best_score:.1f}")
    
    print(f"\n RESUMEN DE OPTIMIZACIÓN BATCH:")
    print("=" * 40)
    for i, res in enumerate(all_results):
        status = " MEJOR" if res['score'] == best_score else ""
        print(f"   Lote {i+1} ({res['seed_range']}): {res['score']:.1f}/100 {status}")
    
    print(f"\n Mejor resultado con score: {best_score:.1f}/100")
    
    return {
        'best_result': best_result,
        'all_results': all_results,
        'best_score': best_score
    }