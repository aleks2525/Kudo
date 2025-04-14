import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import List, Dict, Tuple, Optional

def visualize_pallet_3d(pallet, sku_data_dict):
    """
    Создать 3D визуализацию паллеты на основе оптимизированных слоев
    
    Args:
        pallet: Структура данных паллеты с оптимизированными слоями
        sku_data_dict: Словарь с данными о размерах SKU
        
    Returns:
        Plotly фигура с 3D визуализацией паллеты
    """
    # Стандартные размеры паллеты
    pallet_width = 80
    pallet_length = 120
    pallet_height = 15  # Высота самой паллеты (базы)
    
    # Предопределенная цветовая схема для слоев
    colors = [
        '#1f77b4',  # синий
        '#ff7f0e',  # оранжевый
        '#2ca02c',  # зеленый
        '#d62728',  # красный
        '#9467bd',  # фиолетовый
        '#8c564b',  # коричневый
        '#e377c2',  # розовый
        '#7f7f7f',  # серый
        '#bcbd22',  # оливковый
        '#17becf'   # голубой
    ]
    
    # Создаем 3D визуализацию
    fig = go.Figure()
    
    # Добавляем базу паллеты
    fig.add_trace(go.Mesh3d(
        x=[0, pallet_width, pallet_width, 0, 0, pallet_width, pallet_width, 0],
        y=[0, 0, pallet_length, pallet_length, 0, 0, pallet_length, pallet_length],
        z=[0, 0, 0, 0, pallet_height, pallet_height, pallet_height, pallet_height],
        i=[0, 0, 0, 1, 1, 4],
        j=[1, 2, 3, 2, 3, 5],
        k=[2, 3, 0, 3, 0, 6],
        color='brown',
        opacity=0.5,
        name='Паллета'
    ))
    
    # Устанавливаем начальную высоту над паллетой
    current_z = pallet_height
    
    # Добавляем коробки по слоям
    for layer_idx, layer in enumerate(pallet['layers']):
        layer_height = layer['height']
        layer_color = colors[layer_idx % len(colors)]
        
        # Обходим все коробки в слое
        for box_idx, box in enumerate(layer['boxes']):
            # Получаем позицию и размеры коробки
            if 'positions' in layer and len(layer['positions']) > box_idx:
                x, y, w, l = layer['positions'][box_idx]
            else:
                # Если нет данных о позиции, размещаем в углу
                x, y = 0, 0
                w, l = box.get('width', 30), box.get('length', 30)
            
            # Получаем высоту коробки
            h = box.get('height', layer_height)
            
            # Создаем 3D куб для коробки
            box_vertices = [
                # Нижняя грань
                [x, y, current_z],
                [x+w, y, current_z],
                [x+w, y+l, current_z],
                [x, y+l, current_z],
                # Верхняя грань
                [x, y, current_z+h],
                [x+w, y, current_z+h],
                [x+w, y+l, current_z+h],
                [x, y+l, current_z+h]
            ]
            
            # Индексы вершин для граней куба
            i = [0, 0, 0, 1, 1, 4]
            j = [1, 2, 3, 2, 3, 5]
            k = [2, 3, 0, 3, 0, 6]
            
            # Получаем SKU для текста
            sku = box.get('sku', f"Коробка {box_idx+1}")
            
            # Добавляем куб
            fig.add_trace(go.Mesh3d(
                x=[v[0] for v in box_vertices],
                y=[v[1] for v in box_vertices],
                z=[v[2] for v in box_vertices],
                i=i, j=j, k=k,
                color=layer_color,
                opacity=0.8,
                name=f"{sku} (Слой {layer_idx+1})",
                hoverinfo="text",
                text=f"SKU: {sku}<br>Размеры: {w}x{l}x{h} см<br>Слой: {layer_idx+1}"
            ))
            
            # Добавляем текстовую метку для SKU
            fig.add_trace(go.Scatter3d(
                x=[x + w/2],
                y=[y + l/2],
                z=[current_z + h/2],
                mode='text',
                text=[sku],
                textposition='middle center',
                textfont=dict(size=10, color='black'),
                hoverinfo='none'
            ))
        
        # Увеличиваем текущую высоту для следующего слоя
        current_z += layer_height
    
    # Настройка вида фигуры
    fig.update_layout(
        title="3D визуализация укладки паллеты",
        scene=dict(
            xaxis=dict(title="Ширина (см)", range=[0, pallet_width]),
            yaxis=dict(title="Длина (см)", range=[0, pallet_length]),
            zaxis=dict(title="Высота (см)", range=[0, pallet_height + current_z + 10]),
            aspectmode='data'
        ),
        autosize=True,
        width=800,
        height=700,
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Устанавливаем начальный угол обзора
    fig.update_layout(scene_camera=dict(
        eye=dict(x=1.5, y=-1.5, z=1.2)
    ))
    
    return fig

def display_pallet_visualization_3d(pallet, sku_data_path='data/sku_dimensions.csv'):
    """
    Отобразить 3D визуализацию укладки паллеты с оптимизированными слоями
    
    Args:
        pallet: Структура данных паллеты с заказами и слоями
        sku_data_path: Путь к файлу с данными о SKU
    """
    try:
        # Загружаем данные о размерах SKU
        sku_data = pd.read_csv(sku_data_path)
        sku_data_dict = {row['SKU']: row for _, row in sku_data.iterrows()}
        
        # Проверяем наличие слоев
        if 'layers' in pallet and pallet['layers']:
            # Создаем 3D визуализацию паллеты
            fig = visualize_pallet_3d(pallet, sku_data_dict)
            st.plotly_chart(fig, use_container_width=True)
            
            # Статистика по паллете
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Общая высота паллеты:** {pallet.get('total_height', 0):.1f} см")
                st.write(f"**Общий вес:** {pallet.get('total_weight', 0) / 1000:.2f} кг")
            with col2:
                st.write(f"**Коэффициент использования пространства:** {pallet.get('utilization', 0):.1f}%")
                st.write(f"**Показатель устойчивости:** {pallet.get('stability_score', 0):.1f}/100")
            
            # Отображаем детали слоев
            st.subheader("Детали слоев паллеты")
            for i, layer in enumerate(pallet['layers']):
                st.write(f"**Слой {i+1}**: высота {layer['height']:.1f} см, {len(layer['boxes'])} коробок, общий вес: {layer.get('weight', 0) / 1000:.2f} кг")
                
                # Отображаем информацию о коробках в слое
                box_info = []
                for j, box in enumerate(layer['boxes']):
                    sku = box.get('sku', f"Коробка {j+1}")
                    if 'positions' in layer and len(layer['positions']) > j:
                        x, y, w, l = layer['positions'][j]
                        box_info.append({
                            "SKU": sku,
                            "Размеры (см)": f"{w}x{l}x{box.get('height', layer['height'])}",
                            "Вес (г)": box.get('weight', 0),
                            "Поворот": "Да" if box.get('rotated', False) else "Нет"
                        })
                    else:
                        box_info.append({
                            "SKU": sku,
                            "Размеры (см)": f"{box.get('width', '?')}x{box.get('length', '?')}x{box.get('height', layer['height'])}",
                            "Вес (г)": box.get('weight', 0),
                            "Поворот": "Да" if box.get('rotated', False) else "Нет"
                        })
                
                if box_info:
                    st.dataframe(pd.DataFrame(box_info), use_container_width=True)
        else:
            st.warning("Для этой паллеты недоступна трехмерная визуализация слоев")
    
    except Exception as e:
        st.error(f"Ошибка отображения 3D визуализации паллеты: {str(e)}") 