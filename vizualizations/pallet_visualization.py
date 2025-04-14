import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import List, Dict, Tuple, Optional
import random

class PalletVisualizer:
    """
    Класс для визуализации укладки коробок на паллету
    Позволяет пошагово отображать процесс укладки коробок на паллету
    """
    
    def __init__(self, pallet_size: Tuple[float, float] = (80, 120), max_height: float = 180):
        """
        Инициализация визуализатора паллет
        
        Args:
            pallet_size: Размеры паллеты (ширина x длина) в см
            max_height: Максимальная высота укладки в см
        """
        self.pallet_width, self.pallet_length = pallet_size
        self.max_height = max_height
        
        # Слои паллеты (каждый слой - один SKU)
        self.layers = []
        
        # Цветовая схема для разных SKU (стандартные цвета из Plotly)
        self.colors = [
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
        self.sku_colors = {}
    
    def _get_sku_color(self, sku: str) -> str:
        """Получить цвет для SKU"""
        if sku not in self.sku_colors:
            self.sku_colors[sku] = self.colors[len(self.sku_colors) % len(self.colors)]
        return self.sku_colors[sku]
    
    def add_box(self, sku: str, dimensions: Tuple[float, float, float], weight: float):
        """
        Добавить коробку на паллету
        
        Args:
            sku: Идентификатор товара
            dimensions: Размеры коробки (ширина, длина, высота) в см
            weight: Вес коробки в граммах
        """
        width, length, height = dimensions
        
        # Определяем лучшую ориентацию для коробки
        if width > length:
            width, length = length, width
        
        # Создаем новый слой
        layer = {
            'sku': sku,
            'width': width,
            'length': length,
            'height': height,
            'weight': weight,
            'color': self._get_sku_color(sku),
            'boxes': []  # Коробки этого SKU на данном слое
        }
        
        # Добавляем первую коробку
        box_position = (0, 0)  # Левый нижний угол
        
        layer['boxes'].append({
            'position': box_position,
            'dimensions': (width, length, height)
        })
        
        # Добавляем слой в список слоев
        self.layers.append(layer)
    
    def visualize_pallet(self, step: int = None) -> go.Figure:
        """
        Визуализировать паллету на определенном шаге укладки
        
        Args:
            step: Номер шага (слоя). Если None, показываются все слои
            
        Returns:
            Plotly фигура с визуализацией паллеты
        """
        fig = go.Figure()
        
        # Добавляем контур паллеты
        fig.add_shape(
            type="rect",
            x0=0,
            y0=0,
            x1=self.pallet_width,
            y1=self.pallet_length,
            line=dict(color="black", width=2),
            fillcolor="rgba(200, 200, 200, 0.3)",
            layer="below"
        )
        
        # Если шаг не указан, показываем все слои
        show_layers = self.layers
        if step is not None and 0 <= step < len(self.layers):
            show_layers = self.layers[:step+1]
        
        # Отображаем слои и коробки на них
        current_height = 0
        annotations = []
        
        for i, layer in enumerate(show_layers):
            layer_color = layer['color']
            sku = layer['sku']
            layer_height = layer['height']
            
            # Добавляем коробки текущего слоя
            for box in layer['boxes']:
                x0, y0 = box['position']
                width, length, _ = box['dimensions']
                
                # Добавляем коробку
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    y0=y0,
                    x1=x0 + width,
                    y1=y0 + length,
                    line=dict(color="black", width=1),
                    fillcolor=layer_color,
                    opacity=0.8,
                    layer="below"
                )
                
                # Добавляем информацию о SKU в центре коробки
                annotations.append(
                    dict(
                        x=x0 + width/2,
                        y=y0 + length/2,
                        text=sku,
                        showarrow=False,
                        font=dict(color="black", size=10, family="Arial"),
                    )
                )
            
            current_height += layer_height
        
        # Применяем аннотации
        fig.update_layout(annotations=annotations)
        
        # Настройка осей и общего вида
        fig.update_layout(
            title=f"Визуализация паллеты (шаг {step+1 if step is not None else len(self.layers)} из {len(self.layers)})",
            xaxis=dict(
                title="Ширина (см)",
                range=[-5, self.pallet_width + 5],
                showgrid=True,
                zeroline=True
            ),
            yaxis=dict(
                title="Длина (см)",
                range=[-5, self.pallet_length + 5],
                showgrid=True,
                zeroline=True,
                scaleanchor="x",
                scaleratio=1
            ),
            autosize=True,
            width=600,
            height=500,
            plot_bgcolor='white',
            showlegend=False,
            margin=dict(l=65, r=50, b=65, t=90)
        )
        
        return fig

def create_pallet_visualization(pallet, sku_data_path='data/sku_dimensions.csv'):
    """
    Создать визуализацию укладки паллеты
    
    Args:
        pallet: Структура данных паллеты с заказами
        sku_data_path: Путь к файлу с данными о SKU
        
    Returns:
        Визуализатор паллеты
    """
    # Загружаем данные о размерах SKU
    try:
        sku_data = pd.read_csv(sku_data_path)
    except Exception as e:
        st.error(f"Ошибка загрузки данных о SKU: {str(e)}")
        return None
    
    # Создаем визуализатор паллеты
    visualizer = PalletVisualizer()
    
    # Добавляем коробки для каждого SKU в заказах паллеты
    for order in pallet.get('orders', []):
        order_data = order.get('data', {})
        sku = order_data.get('SKU')
        
        if sku and sku in sku_data['SKU'].values:
            # Получаем размеры SKU
            sku_info = sku_data[sku_data['SKU'] == sku].iloc[0]
            width = sku_info['Ширина (см.)']
            length = sku_info['Длина (см.)']
            height = sku_info['Высота (см.)']
            weight = sku_info['Вес (г.)']
            
            # Добавляем коробку на паллету
            visualizer.add_box(sku, (width, length, height), weight)
    
    return visualizer

def display_pallet_visualization(pallet, sku_data_path='data/sku_dimensions.csv'):
    """
    Отобразить визуализацию укладки паллеты с возможностью пошагового просмотра
    
    Args:
        pallet: Структура данных паллеты с заказами
        sku_data_path: Путь к файлу с данными о SKU
    """
    # Загружаем 3D визуализатор, если он доступен
    try:
        from vizualizations.pallet_visualization_3d import display_pallet_visualization_3d
        
        # Проверяем наличие слоев (для оптимизированных паллет)
        if 'layers' in pallet and pallet['layers']:
            # Используем 3D визуализацию для оптимизированных паллет
            display_pallet_visualization_3d(pallet, sku_data_path)
            return
    except Exception as e:
        st.warning(f"3D визуализация недоступна: {str(e)}")
    
    # Используем стандартную 2D визуализацию, если 3D недоступно или у паллеты нет слоев
    try:
        visualizer = create_pallet_visualization(pallet, sku_data_path)
        
        if not visualizer:
            st.error("Не удалось создать визуализацию паллеты.")
            return
        
        # Создаем состояние для отслеживания текущего шага
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 0
        
        # Определяем максимальное количество шагов
        max_step = len(visualizer.layers) - 1
        
        # Выводим визуализацию паллеты
        fig = visualizer.visualize_pallet(st.session_state.current_step)
        st.plotly_chart(fig, use_container_width=True)
        
        # Кнопки навигации
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("◀ Предыдущий слой", key="prev_step"):
                prev_step()
        
        with col2:
            st.write(f"Слой {st.session_state.current_step + 1} из {max_step + 1}")
        
        with col3:
            if st.button("Следующий слой ▶", key="next_step"):
                next_step()
        
        # Функции для кнопок навигации
        def next_step():
            if st.session_state.current_step < max_step:
                st.session_state.current_step += 1
        
        def prev_step():
            if st.session_state.current_step > 0:
                st.session_state.current_step -= 1
    
    except Exception as e:
        st.error(f"Ошибка отображения визуализации паллеты: {str(e)}")

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