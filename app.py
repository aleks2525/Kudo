import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from vizualizations.warehouse_route import visualize_picking_route
from algorithms.optimized_route import get_optimized_picking_route
from vizualizations.pallet_visualization import display_pallet_visualization
from algorithms.pallet_optimization import optimize_pallet
from collections import defaultdict
import plotly.graph_objects as go

class Warehouse:
    def __init__(self):
        self.layout = None
        self.rows = 0
        self.cols = 0
        self.cell_types = {}
        self.storage_cells = {}  # Changed to dict for easy lookup by cell ID
        self.receiving_cells = set()
        self.shipping_cells = set()
        self.occupied_cells = set()
        self.graph = {}
        self.cell_coords = {}  # Maps cell ID to coordinates
        self.coords_to_cell = {}  # Maps coordinates to cell ID
        
        # Данные о товарах и размещении
        self.sku_dimensions = None
        self.warehouse_data = None
        self.cell_contents = {}
        
        # Данные заявок/заказов
        self.order_data = None

    def load_layout(self, file):
        try:
            df = pd.read_csv(file)
            self.layout = df.values
            rows, cols = self.layout.shape
            
            # Инициализируем соответствие ячеек и их координат
            for i in range(rows):
                for j in range(cols):
                    cell_value = str(self.layout[i, j]).strip()
                    if cell_value and cell_value not in ["nan", "None"]:
                        # Проверяем, похож ли на идентификатор ячейки (например, A01, B12)
                        if len(cell_value) >= 2 and cell_value[0].isalpha() and cell_value[1:].isdigit():
                            self.cell_coords[cell_value] = (i, j)
                            self.coords_to_cell[(i, j)] = cell_value
                            self.storage_cells[cell_value] = (i, j)
            
            self.rows = rows
            self.cols = cols
            return True
        except Exception as e:
            raise Exception(f"Ошибка при загрузке плана склада: {str(e)}")

    def load_sku_dimensions(self, file):
        try:
            self.sku_dimensions = pd.read_csv(file)
            return True
        except Exception as e:
            raise Exception(f"Ошибка при загрузке размеров SKU: {str(e)}")

    def load_warehouse_data(self, file):
        try:
            self.warehouse_data = pd.read_csv(file)
            self._update_cell_contents()
            return True
        except Exception as e:
            raise Exception(f"Ошибка при загрузке данных склада: {str(e)}")

    def _update_cell_contents(self):
        """Обновление информации о содержимом ячеек"""
        self.cell_contents = {}
        if self.warehouse_data is not None:
            for _, row in self.warehouse_data.iterrows():
                if pd.notna(row['SKU']):
                    cell_id = row['Ячейка']
                    # Убедимся, что ячейка в правильном формате
                    self.cell_contents[cell_id] = {
                        'sku': row['SKU'],
                        'quantity': row['Количество'] if pd.notna(row['Количество']) else 0
                    }
                    # Если ячейка содержит товары, добавляем ее в множество занятых ячеек
                    if row['Количество'] > 0:
                        self.occupied_cells.add(cell_id)

    def get_cell_info(self, cell_id: str) -> Dict:
        """Получение информации о содержимом ячейки"""
        if cell_id not in self.cell_contents:
            return None
        
        cell_data = self.cell_contents[cell_id]
        sku_info = None
        
        if self.sku_dimensions is not None and cell_data['sku'] in self.sku_dimensions['SKU'].values:
            sku_info = self.sku_dimensions[
                self.sku_dimensions['SKU'] == cell_data['sku']
            ].iloc[0]
        
        return {
            'sku': cell_data['sku'],
            'quantity': cell_data['quantity'],
            'dimensions': {
                'width': sku_info['Ширина (см.)'] if sku_info is not None else None,
                'length': sku_info['Длина (см.)'] if sku_info is not None else None,
                'height': sku_info['Высота (см.)'] if sku_info is not None else None,
                'weight': sku_info['Вес (г.)'] if sku_info is not None else None
            } if sku_info is not None else None
        }

    def get_warehouse_stats(self) -> Dict:
        try:
            total_cells = len(self.storage_cells)
            occupied_cells = len(self.occupied_cells)
            
            return {
                "status": "активен",
                "dimensions": f"{self.rows}x{self.cols}",
                "total_cells": total_cells,
                "storage_cells": total_cells,
                "occupied_cells": occupied_cells,
                "occupancy": (occupied_cells / total_cells * 100) if total_cells > 0 else 0.0
            }
        except Exception as e:
            return {
                "status": "ошибка",
                "message": str(e),
                "dimensions": "0x0",
                "total_cells": 0,
                "storage_cells": 0,
                "occupied_cells": 0,
                "occupancy": 0.0
            }

    def load_order_data(self, file):
        """Загрузка данных о заявках/заказах"""
        try:
            self.order_data = pd.read_csv(file)
            return True
        except Exception as e:
            raise Exception(f"Ошибка при загрузке данных о заявках: {str(e)}")
            
    def get_order_stats(self) -> Dict:
        """Получение статистики по заявкам"""
        if self.order_data is None:
            return {
                "status": "не загружено",
                "total_orders": 0
            }
            
        try:
            total_orders = len(self.order_data)
            
            # Собираем дополнительную статистику
            stats = {
                "status": "загружено",
                "total_orders": total_orders
            }
            
            # Если в данных есть колонка Статус, добавляем статистику по статусам
            if 'Статус' in self.order_data.columns:
                status_counts = self.order_data['Статус'].value_counts().to_dict()
                stats["status_counts"] = status_counts
                
            # Если есть колонка Дата, добавляем статистику по датам
            if 'Дата' in self.order_data.columns:
                stats["dates_range"] = (
                    self.order_data['Дата'].min(),
                    self.order_data['Дата'].max()
                )
                
            return stats
        except Exception as e:
            return {
                "status": "ошибка",
                "message": str(e),
                "total_orders": 0
            }

    def update_cell_quantity(self, cell_id: str, quantity_change: int) -> bool:
        """
        Обновляет количество товара в ячейке
        
        Args:
            cell_id: ID ячейки
            quantity_change: Изменение количества (отрицательное для уменьшения)
            
        Returns:
            bool: True, если обновление прошло успешно
        """
        if cell_id not in self.cell_contents:
            return False
            
        current_quantity = self.cell_contents[cell_id]['quantity']
        new_quantity = current_quantity + quantity_change
        
        # Проверяем, что новое количество неотрицательное
        if new_quantity < 0:
            return False
            
        # Обновляем количество
        self.cell_contents[cell_id]['quantity'] = new_quantity
        
        # Обновляем список занятых ячеек
        if new_quantity > 0:
            self.occupied_cells.add(cell_id)
        else:
            if cell_id in self.occupied_cells:
                self.occupied_cells.remove(cell_id)
                
        return True
        
    def save_warehouse_data(self, file_path: str = 'data/warehouse_data.csv'):
        """
        Сохраняет текущие данные склада в CSV-файл
        
        Args:
            file_path: Путь к файлу для сохранения
        
        Returns:
            bool: True, если сохранение прошло успешно
        """
        try:
            # Создаем DataFrame из текущих данных cell_contents
            data = []
            for cell_id, contents in self.cell_contents.items():
                data.append({
                    'Ячейка': cell_id,
                    'SKU': contents['sku'],
                    'Количество': contents['quantity']
                })
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            return True
        except Exception as e:
            print(f"Ошибка при сохранении данных склада: {str(e)}")
            return False

class WarehouseApp:
    def __init__(self):
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
        if 'current_view' not in st.session_state:
            st.session_state.current_view = 'main'
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        if 'operators' not in st.session_state:
            st.session_state.operators = {}
        if 'selected_cell' not in st.session_state:
            st.session_state.selected_cell = None
        if 'action' not in st.session_state:
            st.session_state.action = None
        if 'pallets' not in st.session_state:
            st.session_state.pallets = {}  # Dictionary to store pallets assigned to operators
        if 'orders_in_pallets' not in st.session_state:
            st.session_state.orders_in_pallets = set()  # Track which orders are already in pallets
        if 'distributed_boxes' not in st.session_state:
            st.session_state.distributed_boxes = {}  # Track distributed boxes by order_id and SKU

        self.warehouse = Warehouse()
        
        # Автоматическая загрузка данных из файлов при запуске
        self.load_initial_data()

        if 'settings' not in st.session_state:
            st.session_state.settings = {
                # Настройки паллеты
                'pallet_extension': 3.0,  # см
                'pallet_load_height': 200.0,  # см (2 м)
                'pallet_weight': 1000.0,  # кг
                'bucket_weight': 500.0,  # г
                'pallet_stability': 10.0,  # %
                'pallet_height_diff': 20.0,  # см
                'fill_complex_gaps': False,  # Заполнение пустот нестандартной формы
                
                # Репорты
                'report_time_at_pallet': True,
                'report_time_threshold': 5.0,  # минут
                'report_operator_pallet': True
            }

    def load_initial_data(self):
        """Загрузка начальных данных из файлов"""
        try:
            # Создаем каталог data, если он не существует
            import os
            if not os.path.exists("data"):
                os.makedirs("data")
                # Не показываем уведомление при создании каталога
            
            # Создаем пример файла с данными о заявках, если он не существует
            order_file = Path("data/order_data.csv")
            if not order_file.exists():
                # Создаем пример данных о заявках
                sample_orders = pd.DataFrame({
                    'SKU': ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005'],
                    'Количество': [25, 30, 22, 18, 27],
                    'Статус': ['новый', 'новый', 'новый', 'новый', 'новый'],
                    'Дата': ['2023-10-01', '2023-10-01', '2023-10-02', '2023-10-02', '2023-10-03']
                })
                sample_orders.to_csv(order_file, index=False)
                # Не показываем уведомление при создании примера
            
            # Проверяем наличие файлов и загружаем данные
            sklad_file = Path("data/sklad.csv")
            if sklad_file.exists():
                self.warehouse.load_layout(sklad_file)
                # Не показываем уведомление при автозагрузке данных
                st.session_state.initialized = True
            
            sku_file = Path("data/sku_dimensions.csv")
            if sku_file.exists():
                self.warehouse.load_sku_dimensions(sku_file)
                # Не показываем уведомление при автозагрузке данных
            
            wh_data_file = Path("data/warehouse_data.csv")
            if wh_data_file.exists():
                self.warehouse.load_warehouse_data(wh_data_file)
                # Не показываем уведомление при автозагрузке данных
                
            # Загружаем данные о заявках
            if order_file.exists():
                self.warehouse.load_order_data(order_file)
                # Не показываем уведомление при автозагрузке данных
        except Exception as e:
            # Показываем только сообщения об ошибках
            self.add_notification(f"Ошибка при загрузке данных: {str(e)}", "error")
            import traceback
            print(traceback.format_exc())
            
    def get_distribution_stats(self):
        """
        Gets consistent statistics about distributed orders and boxes
        
        Returns:
            Dict with statistics: total_orders, total_boxes, distributed_orders, distributed_boxes
        """
        # Get total order statistics from the original data
        total_orders = 0
        total_boxes = 0
        if self.warehouse.order_data is not None:
            total_orders = len(self.warehouse.order_data)
            if 'Количество' in self.warehouse.order_data.columns:
                total_boxes = int(self.warehouse.order_data['Количество'].sum())
        
        # Get distributed statistics from pallets
        distributed_orders = len(st.session_state.orders_in_pallets)
        distributed_boxes = 0
        for op_id, pallets in st.session_state.pallets.items():
            for pallet in pallets:
                distributed_boxes += sum(order.get('box_count', 0) for order in pallet.get('orders', []))
        
        return {
            'total_orders': total_orders,
            'total_boxes': total_boxes,
            'distributed_orders': distributed_orders,
            'distributed_boxes': distributed_boxes,
            'distribution_percent': (distributed_boxes / total_boxes * 100) if total_boxes > 0 else 0
        }

    def add_notification(self, message: str, type: str = "info"):
        """
        Add a notification, but only show it if it's an error or warning
        """
        # Для действий операторов и загрузки данных не показываем уведомления
        silent_notifications = [
            "План склада загружен успешно",
            "Данные о размерах товаров загружены",
            "Данные склада загружены", 
            "Данные о заявках загружены",
            "Оператор",
            "Задача",
            "Паллета",
            "Начата комплектация",
            "Выберите целевую ячейку",
            "Запущена инвентаризация",
            "Настройки успешно сохранены",
            "Все артикулы уже распределены"
        ]
        
        # Если это системное уведомление, не отображаем его
        for silent_msg in silent_notifications:
            if silent_msg in message and type != "error":
                return  # не добавляем уведомление
        
        # Добавляем только важные уведомления и ошибки
        notification_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        st.session_state.notifications.append({
            "id": notification_id,
            "message": message,
            "type": type,
            "time": datetime.now()
        })

    def remove_notification(self, notification_id: str):
        st.session_state.notifications = [
            n for n in st.session_state.notifications 
            if n["id"] != notification_id
        ]

    def show_notifications(self):
        # Удаляем фильтрацию по времени истечения
        # Отображаем только последние 5 уведомлений для экономии места
        for note in st.session_state.notifications[-5:]:
            col1, col2 = st.columns([20, 1])
            with col1:
                if note["type"] == "error":
                    st.error(note["message"])
                elif note["type"] == "warning":
                    st.warning(note["message"])
                elif note["type"] == "success":
                    st.success(note["message"])
                else:
                    st.info(note["message"])
            with col2:
                if st.button("×", key=f"close_{note['id']}"):
                    self.remove_notification(note["id"])

    def run(self):
        st.title("Оптимизация процесса паллетизации товаров")
        
        self.show_notifications()
        self.show_sidebar()
        
        if st.session_state.current_view == 'main':
            self.show_main_view()
        elif st.session_state.current_view == 'admin':
            self.show_admin_view()
        elif st.session_state.current_view == 'operators':
            self.show_operators_view()
        elif st.session_state.current_view == 'statistics':
            self.show_statistics_view()
        elif st.session_state.current_view == 'settings':
            self.show_settings_view()

    def show_sidebar(self):
        with st.sidebar:
            st.title("Навигация")
            
            if st.button("🏠 Главная", key="nav_main"):
                st.session_state.current_view = 'main'
            if st.button("👨‍💼 Администратор", key="nav_admin"):
                st.session_state.current_view = 'admin'
            if st.button("👥 Операторы", key="nav_operators"):
                st.session_state.current_view = 'operators'
            if st.button("📊 Статистика", key="nav_stats"):
                st.session_state.current_view = 'statistics'
            if st.button("⚙️ Настройки", key="nav_settings"):
                st.session_state.current_view = 'settings'
                
            st.divider()
            
            # Загрузка файлов
            st.subheader("Загрузка данных")
            
            sklad_file = st.file_uploader("План склада (sklad.csv)", type=['csv'])
            if sklad_file:
                try:
                    self.warehouse.load_layout(sklad_file)
                    self.add_notification("План склада загружен успешно", "success")
                    st.session_state.initialized = True
                except Exception as e:
                    self.add_notification(str(e), "error")

            sku_file = st.file_uploader("Размеры товаров (sku_dimensions.csv)", type=['csv'])
            if sku_file:
                try:
                    self.warehouse.load_sku_dimensions(sku_file)
                    self.add_notification("Данные о размерах товаров загружены", "success")
                except Exception as e:
                    self.add_notification(str(e), "error")

            wh_data_file = st.file_uploader("Данные склада (warehouse_data.csv)", type=['csv'])
            if wh_data_file:
                try:
                    self.warehouse.load_warehouse_data(wh_data_file)
                    self.add_notification("Данные склада загружены", "success")
                except Exception as e:
                    self.add_notification(str(e), "error")
            
            # Добавляем загрузку файла заявок
            order_file = st.file_uploader("Данные заявки (order_data.csv)", type=['csv'])
            if order_file:
                try:
                    self.warehouse.load_order_data(order_file)
                    self.add_notification("Данные о заявках загружены", "success")
                except Exception as e:
                    self.add_notification(str(e), "error")
            
            st.divider()
            
            # Статистика склада
            st.write("Статистика склада:")
            stats = self.warehouse.get_warehouse_stats()
            
            if stats["status"] == "активен":
                st.write(f"Размеры: {stats['dimensions']}")
                st.write(f"Всего ячеек: {stats['total_cells']}")
                st.write(f"Занято ячеек: {stats['occupied_cells']}")
                st.write(f"Загруженность: {stats['occupancy']:.1f}%")
                
                # Отображаем процентное соотношение в виде прогресс-бара
                st.progress(min(stats['occupancy'] / 100, 1.0))
            else:
                st.write(f"Статус: {stats['status']}")
                if 'message' in stats:
                    st.write(f"Ошибка: {stats['message']}")

    def show_main_view(self):
        st.header("Обзор склада")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("План склада")
            if self.warehouse and self.warehouse.layout is not None:
                st.write("Условные обозначения:")
                st.write("📦 - Пустая ячейка")
                st.write("🟩 - Заполненная ячейка")
                st.write("🔄 - Точка старта/финиша")
                
                # Определяем точку старта/финиша в левом нижнем углу
                start_finish_point = (self.warehouse.rows - 1, 1)  # Нижний левый угол
                
                # Если активен режим установки точки старта/финиша
                if st.session_state.action == "set_start_finish" and st.session_state.selected_cell:
                    cell_id = st.session_state.selected_cell
                    if cell_id in self.warehouse.cell_coords:
                        st.session_state.start_finish_point = self.warehouse.cell_coords[cell_id]
                        self.add_notification(f"Точка старта/финиша установлена в ячейке {cell_id}", "success")
                        st.session_state.action = None
                        st.rerun()
                
                # Отображаем план склада как таблицу HTML
                html_table = '<div style="font-family: monospace;">'
                html_table += '<table style="border-collapse: collapse;">'
                
                for i in range(self.warehouse.rows):
                    html_table += '<tr>'
                    for j in range(self.warehouse.cols):
                        cell_pos = (i, j)
                        cell_id = self.warehouse.coords_to_cell.get(cell_pos, '')
                        
                        # Получаем содержимое ячейки
                        cell_content = str(self.warehouse.layout[i, j])
                        cell_content = cell_content.strip() if cell_content != 'nan' else ''
                        
                        # Определяем, является ли ячейка точкой старта/финиша
                        is_start_finish = cell_pos == start_finish_point
                        
                        # Определяем стиль ячейки
                        if is_start_finish:
                            # Точка старта/финиша
                            bg_color = "#ffcc99"  # Оранжевый для точки старта/финиша
                            display_content = "🔄"
                            td_style = f'background-color: {bg_color}; font-weight: bold;'
                        elif cell_id and cell_id in self.warehouse.storage_cells:
                            # Это складская ячейка
                            is_occupied = cell_id in self.warehouse.cell_contents and self.warehouse.cell_contents[cell_id]['quantity'] > 0
                            
                            if is_occupied:
                                bg_color = "#c8e6c9"  # Светло-зеленый для занятых ячеек
                                quantity = self.warehouse.cell_contents[cell_id]['quantity']
                                display_content = f"{cell_id} ({quantity})"
                            else:
                                bg_color = "#f5f5f5"  # Светло-серый для пустых ячеек
                                display_content = cell_id
                            
                            # Создаем ячейку с кнопкой
                            button_key = f"cell_{i}_{j}"
                            if st.button(display_content, key=button_key):
                                st.session_state.selected_cell = cell_id
                                
                            td_style = f'background-color: {bg_color};'
                        else:
                            # Это не складская ячейка
                            if cell_content == '':
                                bg_color = "#ffffff"  # Белый для пустых мест
                                display_content = "&nbsp;"
                            else:
                                # Любое другое содержимое
                                bg_color = "#e0e0e0"  # Серый для других элементов
                                display_content = cell_content
                            
                            td_style = f'background-color: {bg_color};'
                            
                        # Добавляем ячейку в HTML
                        html_table += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center; {td_style}">{display_content}</td>'
                    
                    html_table += '</tr>'
                
                html_table += '</table></div>'
                
                # Отображаем HTML таблицу с возможностью прокрутки
                st.components.v1.html(
                    f'<div style="max-height: 500px; overflow-y: auto; overflow-x: auto;">{html_table}</div>', 
                    height=500
                )
            
        with col2:
            st.subheader("Информация о ячейке")
            if st.session_state.selected_cell:
                cell_info = self.warehouse.get_cell_info(st.session_state.selected_cell)
                st.write(f"**Ячейка:** {st.session_state.selected_cell}")
                
                if cell_info:
                    st.write(f"**SKU:** {cell_info['sku']}")
                    st.write(f"**Количество:** {cell_info['quantity']} ед.")
                    
                    if cell_info['dimensions']:
                        with st.expander("Характеристики товара"):
                            st.write(f"**Размеры товара:**")
                            st.write(f"- Ширина: {cell_info['dimensions']['width']} см")
                            st.write(f"- Длина: {cell_info['dimensions']['length']} см")
                            st.write(f"- Высота: {cell_info['dimensions']['height']} см")
                            st.write(f"- Вес: {cell_info['dimensions']['weight']} г")
                else:
                    st.write("Ячейка пуста")
                
                # Добавляем действия с ячейкой
                st.divider()
                st.subheader("Действия")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Переместить товар", key="move_btn"):
                        st.session_state.action = "move"
                        self.add_notification(f"Выберите целевую ячейку для перемещения", "info")
                with col_b:
                    if st.button("Инвентаризация", key="inventory_btn"):
                        self.add_notification(f"Запущена инвентаризация ячейки {st.session_state.selected_cell}", "info")
            else:
                st.write("Выберите ячейку на плане склада")

    def show_admin_view(self):
        st.header("Панель администратора")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Добавить оператора")
            with st.form("add_operator"):
                name = st.text_input("ФИО оператора")
                role = st.selectbox("Роль", ["Водитель погрузчика", "Контролер"])
                if st.form_submit_button("Добавить"):
                    if name:
                        operator_id = len(st.session_state.operators)
                        st.session_state.operators[operator_id] = {
                            'name': name,
                            'role': role,
                            'status': 'не работает',
                            'added_time': datetime.now(),
                            'tasks': []
                        }
                        self.add_notification(f"Оператор {name} добавлен", "success")
                    else:
                        self.add_notification("Введите ФИО оператора", "warning")
            
            # Добавляем статистику по заявкам
            st.subheader("Статистика артикулов")
            order_stats = self.warehouse.get_order_stats()
            
            if order_stats["status"] == "загружено":
                # Получаем общую и распределенную статистику
                dist_stats = self.get_distribution_stats()
                
                st.write(f"**Общее количество артикулов:** {dist_stats['total_orders']}")
                st.write(f"**Общее количество коробок:** {dist_stats['total_boxes']}")
                
                # Если есть данные о статусах, отображаем их
                if "status_counts" in order_stats:
                    st.write("**Статусы артикулов:**")
                    for status, count in order_stats["status_counts"].items():
                        st.write(f"- {status}: {count}")
                
                # Если есть данные о датах, отображаем их
                if "dates_range" in order_stats:
                    st.write(f"**Период:** с {order_stats['dates_range'][0]} по {order_stats['dates_range'][1]}")
                
                # Если есть данные о заявках, отображаем их в таблице
                if self.warehouse.order_data is not None:
                    with st.expander("Детали артикулов"):
                        # Добавляем веса из sku_dimensions.csv
                        order_data_with_weights = self.warehouse.order_data.copy()
                        
                        # Если есть колонка SKU в данных заявки и загружены данные о размерах SKU
                        if 'SKU' in order_data_with_weights.columns and self.warehouse.sku_dimensions is not None:
                            # Объединяем данные по полю SKU
                            order_data_with_weights = pd.merge(
                                order_data_with_weights,
                                self.warehouse.sku_dimensions[['SKU', 'Вес (г.)', 'Высота (см.)']],
                                on='SKU',
                                how='left'
                            )
                            
                            # Добавляем колонку с общим весом (вес единицы * количество)
                            if 'Количество' in order_data_with_weights.columns and 'Вес (г.)' in order_data_with_weights.columns:
                                order_data_with_weights['Общий вес (кг)'] = order_data_with_weights['Количество'] * order_data_with_weights['Вес (г.)'] / 1000
                        
                        st.dataframe(order_data_with_weights)
            else:
                st.write("Данные об артикулах не загружены")
            
            # Добавляем кнопку отправки заявки операторам
            st.subheader("Управление задачами")
            col_tasks_1, col_tasks_2 = st.columns(2)
            
            with col_tasks_2:
                # Кнопка распределения заявок по паллетам активна, если есть заявки и доступные операторы
                has_active_operators = any(op['status'] == 'готов к работе' for op_id, op in st.session_state.operators.items())
                has_orders = self.warehouse.order_data is not None and len(self.warehouse.order_data) > 0
                
                # Создаем кнопку с подсветкой если условия соблюдены
                button_label = "Распределить артикулы"
                button_help = ""
                
                if not has_active_operators:
                    button_help += "Нет доступных операторов. "
                if not has_orders:
                    button_help += "Нет загруженных артикулов. "
                
                button_disabled = not (has_active_operators and has_orders)
                
                if st.button(button_label, disabled=button_disabled, help=button_help):
                    self.distribute_orders_to_pallets()
                    
                # Если есть распределенные паллеты, показываем статистику
                pallets_count = sum(len(pallets) for op_id, pallets in st.session_state.pallets.items())
                if pallets_count > 0:
                    with st.expander("Статистика распределения"):
                        # Получаем согласованную статистику
                        dist_stats = self.get_distribution_stats()
                        
                        st.write(f"**Всего паллет:** {pallets_count}")
                        st.write(f"**Распределено артикулов:** {dist_stats['distributed_orders']} из {dist_stats['total_orders']}")
                        st.write(f"**Распределено коробок:** {dist_stats['distributed_boxes']} из {dist_stats['total_boxes']} ({dist_stats['distribution_percent']:.1f}%)")
                        
                        # Проверяем на несоответствие и выводим предупреждение
                        if dist_stats['distributed_boxes'] < dist_stats['total_boxes']:
                            st.warning(f"⚠️ Не все коробки распределены. Осталось распределить {dist_stats['total_boxes'] - dist_stats['distributed_boxes']} коробок.")
                        
                        # Статистика по операторам
                        st.write("**Распределение по операторам:**")
                        for op_id, pallets in st.session_state.pallets.items():
                            if pallets:
                                op_name = st.session_state.operators[op_id]['name']
                                op_pallets = len(pallets)
                                op_boxes = sum(
                                    sum(order.get('box_count', 0) for order in pallet.get('orders', []))
                                    for pallet in pallets
                                )
                                st.write(f"- {op_name}: {op_pallets} паллет, {op_boxes} коробок")
        
        with col2:
            st.subheader("Управление операторами")
            if st.session_state.operators:
                for op_id, operator in st.session_state.operators.items():
                    with st.expander(f"{operator['name']} ({operator['role']}) - {operator['status']}"):
                        st.write(f"Статус: {operator['status']}")
                        st.write(f"Добавлен: {operator['added_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Добавляем кнопки для управления оператором
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("Посмотреть выполнение", key=f"view_perf_{op_id}"):
                                self.add_notification(f"Просмотр выполнения работы оператора {operator['name']}", "info")
                        with col_b:
                            if st.button("Удалить оператора", key=f"del_op_{op_id}"):
                                del st.session_state.operators[op_id]
                                self.add_notification(f"Оператор {operator['name']} удален", "info")
                        
                        # Отображаем задачи оператора
                        if 'tasks' in operator and operator['tasks']:
                            st.write("Текущие задачи:")
                            for task in operator['tasks']:
                                st.write(f"- {task['description']} ({task['status']})")
                        else:
                            st.write("Нет активных задач")
                            
                        # Отображаем паллеты оператора, если они есть
                        if op_id in st.session_state.pallets and st.session_state.pallets[op_id]:
                            st.write(f"**Паллеты оператора ({len(st.session_state.pallets[op_id])}):**")
                            for i, pallet in enumerate(st.session_state.pallets[op_id], 1):
                                orders_count = len(pallet.get('orders', []))
                                boxes_count = pallet.get('box_count', 0)
                                pallet_height = pallet.get('total_height', 0)
                                st.write(f"- Паллета {i}: {orders_count} артикулов, {boxes_count} коробок, высота: {pallet_height:.1f} см ({pallet['status']})")
            else:
                st.write("Нет активных операторов")

    def show_operators_view(self):
        st.header("Список операторов")
        
        if st.session_state.operators:
            st.subheader("Текущие операторы")
            
            # Создаем таблицу операторов
            operators_data = []
            for op_id, operator in st.session_state.operators.items():
                operators_data.append({
                    "ID": op_id,
                    "ФИО": operator['name'],
                    "Должность": operator['role'],
                    "Статус": operator['status']
                })
            
            if operators_data:
                operators_df = pd.DataFrame(operators_data)
                st.dataframe(operators_df, use_container_width=True)
                
                # Отображаем действия для каждого оператора
                st.subheader("Действия оператора")
                selected_operator_name = st.selectbox(
                    "Выберите оператора:",
                    options=[op["ФИО"] for op in operators_data],
                    index=0 if operators_data else None
                )
                
                if selected_operator_name:
                    # Находим ID оператора по имени
                    selected_op_id = None
                    selected_operator = None
                    for op_id, op in st.session_state.operators.items():
                        if op['name'] == selected_operator_name:
                            selected_op_id = op_id
                            selected_operator = op
                            break
                    
                    if selected_operator:
                        st.write(f"Выбран оператор: {selected_operator_name}")
                        st.write(f"Текущий статус: {selected_operator['status']}")
                        
                        # Действия оператора
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Начать работу", key=f"start_work_{selected_op_id}"):
                                # Меняем статус оператора
                                selected_operator['status'] = 'готов к работе'
                                self.add_notification(f"Оператор {selected_operator_name} начал работу", "success")
                                st.rerun()
                        with col2:
                            if st.button("Закончить работу", key=f"end_work_{selected_op_id}"):
                                # Меняем статус оператора
                                selected_operator['status'] = 'не работает'
                                self.add_notification(f"Оператор {selected_operator_name} закончил работу", "info")
                                st.rerun()
                        
                        # Отображаем задачи оператора
                        if 'tasks' in selected_operator and selected_operator['tasks']:
                            st.subheader("Текущие задачи")
                            for task in selected_operator['tasks']:
                                with st.container():
                                    st.write(f"**{task['description']}**")
                                    st.write(f"Статус: {task['status']}")
                                    st.write(f"Создана: {task['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                                    if task['status'] == 'новая':
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            if st.button("Принять задачу", key=f"accept_{task['id']}"):
                                                task['status'] = 'в работе'
                                                self.add_notification(f"Задача {task['description']} принята", "success")
                                                st.rerun()
                                        with col_b:
                                            if st.button("Отклонить задачу", key=f"reject_{task['id']}"):
                                                task['status'] = 'отклонена'
                                                self.add_notification(f"Задача {task['description']} отклонена", "warning")
                                                st.rerun()
                                    elif task['status'] == 'в работе':
                                        if st.button("Завершить задачу", key=f"complete_{task['id']}"):
                                            task['status'] = 'выполнена'
                                            self.add_notification(f"Задача {task['description']} выполнена", "success")
                                            st.rerun()
                        # Удаляем сообщение "Нет активных задач"
                        # else:
                        #     st.info("Нет активных задач")
                            
                        # Отображаем паллеты оператора, если они есть
                        if selected_op_id in st.session_state.pallets and st.session_state.pallets[selected_op_id]:
                            st.subheader("Паллеты для комплектации")
                            for pallet in st.session_state.pallets[selected_op_id]:
                                with st.expander(f"Паллета {pallet['id']} ({pallet['status']})"):
                                    st.write(f"**Создана:** {pallet['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                                    st.write(f"**Статус:** {pallet['status']}")
                                    
                                    # Отображаем общую информацию о паллете
                                    box_count = pallet.get('box_count', sum(order.get('data', {}).get('Количество', 1) for order in pallet.get('orders', [])))
                                    total_weight = pallet.get('total_weight', 0) / 1000  # переводим в кг
                                    total_volume = pallet.get('total_volume', 0) / 1000000  # переводим в м³
                                    total_height = pallet.get('total_height', 0)  # высота в см
                                    
                                    st.write(f"**Всего коробок:** {box_count}")
                                    if total_weight > 0:
                                        st.write(f"**Общий вес:** {total_weight:.2f} кг")
                                    if total_volume > 0:
                                        st.write(f"**Общий объем:** {total_volume:.2f} м³")
                                    st.write(f"**Высота паллеты:** {total_height:.1f} см")
                                    
                                    # Отображаем информацию о слоях, если она есть
                                    if 'layers' in pallet and pallet['layers']:
                                        st.markdown("##### Структура паллеты по слоям:")
                                        layers_info = []
                                        for i, layer in enumerate(pallet['layers'], 1):
                                            layers_info.append(f"- Слой {i}: высота {layer['height']:.1f} см, {layer['boxes']} коробок")
                                        st.markdown("\n".join(layers_info))
                                    
                                    # Проверяем, не превышена ли максимальная высота
                                    max_height = 200  # максимальная высота 2м (200 см)
                                    if total_height > max_height:
                                        st.warning(f"⚠️ Превышена максимальная высота паллеты ({max_height} см)")
                                    
                                    # Отображаем артикулы в паллете
                                    st.write("**Артикулы в паллете:**")
                                    for i, order in enumerate(pallet['orders'], 1):
                                        order_data = order.get('data', {})
                                        sku = order_data.get('SKU', 'Неизвестный SKU')
                                        quantity = order.get('box_count', order_data.get('Количество', 1))
                                        height = order.get('height', 0)
                                        
                                        placed_in_layers = order.get('placed_in_layers', False)
                                        layer_info = " (размещено с учетом слоев)" if placed_in_layers else ""
                                        
                                        st.write(f"**{i}. Артикул {sku}, Количество: {quantity} шт., Высота коробки: {height:.1f} см{layer_info}**")
                                        
                                        # Отображаем данные артикула в виде таблицы с актуальным количеством
                                        modified_order_data = order_data.copy()
                                        # Заменяем количество в данных на реальное количество на паллете
                                        modified_order_data['Количество'] = quantity
                                        order_df = pd.DataFrame([modified_order_data])
                                        st.dataframe(order_df)
                                    
                                    # Кнопка "Приступить к комплектации" (активна только если паллета новая)
                                    if pallet['status'] == 'новая':
                                        if st.button("Приступить к комплектации", key=f"start_pallet_{pallet['id']}"):
                                            # Меняем статус паллеты
                                            pallet['status'] = 'в работе'
                                            # Добавляем информацию о времени начала комплектации
                                            pallet['started_at'] = datetime.now()
                                            self.add_notification(f"Начата комплектация паллеты {pallet['id']}", "success")
                                            st.rerun()
                                    elif pallet['status'] == 'в работе':
                                        # Отображаем маршрут комплектации
                                        st.subheader("Маршрут комплектации")
                                        
                                        # Отображаем время начала комплектации
                                        if 'started_at' in pallet:
                                            time_elapsed = datetime.now() - pallet['started_at']
                                            elapsed_minutes = time_elapsed.total_seconds() / 60
                                            st.write(f"Время комплектации: {elapsed_minutes:.1f} мин.")
                                        
                                        # Используем оптимизированный алгоритм маршрутизации
                                        with st.spinner("Рассчитываем оптимальный маршрут..."):
                                            route_data = get_optimized_picking_route(self.warehouse, pallet)
                                        
                                        if route_data and 'visualization' in route_data:
                                            # Создаем табы для переключения между визуализациями
                                            tab_route, tab_pallet = st.tabs(["Маршрут по складу", "Визуализация паллеты"])
                                            
                                            with tab_route:
                                                # Отображаем информацию о порядке обхода (от тяжелого к легкому)
                                                if 'weights' in route_data and route_data['weights']:
                                                    st.write("### Инструкция комплектации")
                                                    order_info = []
                                                    for cell_id in route_data['cells']:
                                                        # Находим SKU в этой ячейке
                                                        cell_contents = self.warehouse.cell_contents.get(cell_id, {})
                                                        sku = cell_contents.get('sku', 'Неизвестный SKU')
                                                        weight = route_data['weights'].get(sku, 0)
                                                        
                                                        # Получаем количество коробок из заказа
                                                        quantity = 0
                                                        for order in pallet['orders']:
                                                            if order.get('data', {}).get('SKU') == sku:
                                                                # Используем box_count вместо data.Количество для согласованности с описанием
                                                                quantity = order.get('box_count', order.get('data', {}).get('Количество', 1))
                                                                break
                                                        
                                                        order_info.append({
                                                            "Ячейка": cell_id,
                                                            "SKU": sku,
                                                            "Количество": quantity,
                                                            "Вес (г)": weight,
                                                            "Общий вес (кг)": (weight * quantity) / 1000
                                                        })
                                                    
                                                    if order_info:
                                                        df = pd.DataFrame(order_info)
                                                        st.dataframe(df, use_container_width=True)
                                                
                                                # Отображаем график маршрута
                                                st.plotly_chart(route_data['visualization'], use_container_width=True)
                                            
                                            with tab_pallet:
                                                display_pallet_visualization(pallet)
                                        else:
                                            # Если оптимизированный маршрут не сработал, используем стандартный
                                            st.warning("Не удалось построить оптимизированный маршрут, используем стандартный")
                                            route_fig = visualize_picking_route(self.warehouse, pallet)
                                            
                                            if route_fig:
                                                # Создаем табы для переключения между визуализациями
                                                tab_route, tab_pallet = st.tabs(["Маршрут по складу", "Визуализация паллеты"])
                                                
                                                with tab_route:
                                                    st.plotly_chart(route_fig, use_container_width=True)
                                                
                                                with tab_pallet:
                                                    display_pallet_visualization(pallet)
                                            else:
                                                st.warning("Не удалось построить маршрут комплектации")
                                        
                                        if st.button("Завершить комплектацию", key=f"complete_pallet_{pallet['id']}"):
                                            # Меняем статус паллеты
                                            pallet['status'] = 'собрана'
                                            # Добавляем информацию о времени завершения комплектации
                                            pallet['completed_at'] = datetime.now()
                                            
                                            # Уменьшаем количество товаров в ячейках
                                            if route_data and 'cells' in route_data:
                                                updated_cells = []
                                                for cell_id in route_data['cells']:
                                                    # Уменьшаем количество товара в ячейке на 1
                                                    if self.warehouse.update_cell_quantity(cell_id, -1):
                                                        updated_cells.append(cell_id)
                                                
                                                # Сохраняем обновленные данные склада
                                                self.warehouse.save_warehouse_data()
                                                
                                                # Добавляем информацию об отгруженных товарах
                                                if updated_cells:
                                                    pallet['picked_cells'] = updated_cells
                                            else:
                                                # Для стандартного маршрута: получаем список ячеек из заказов
                                                cell_ids = []
                                                for order in pallet.get('orders', []):
                                                    order_data = order.get('data', {})
                                                    sku = order_data.get('SKU')
                                                    
                                                    # Находим ячейку с этим SKU
                                                    for cell_id, contents in self.warehouse.cell_contents.items():
                                                        if contents.get('sku') == sku and contents.get('quantity', 0) > 0:
                                                            cell_ids.append(cell_id)
                                                            break
                                                
                                                # Уменьшаем количество товаров в найденных ячейках
                                                updated_cells = []
                                                for cell_id in cell_ids:
                                                    # Уменьшаем количество товара в ячейке на 1
                                                    if self.warehouse.update_cell_quantity(cell_id, -1):
                                                        updated_cells.append(cell_id)
                                                
                                                # Сохраняем обновленные данные склада
                                                self.warehouse.save_warehouse_data()
                                                
                                                # Добавляем информацию об отгруженных товарах
                                                if updated_cells:
                                                    pallet['picked_cells'] = updated_cells
                                            
                                            # Рассчитываем время комплектации
                                            if 'started_at' in pallet:
                                                completion_time = pallet['completed_at'] - pallet['started_at']
                                                pallet['completion_time'] = completion_time.total_seconds() / 60  # в минутах
                                                self.add_notification(f"Паллета {pallet['id']} собрана за {pallet['completion_time']:.1f} мин.", "success")
                                            else:
                                                self.add_notification(f"Паллета {pallet['id']} собрана", "success")
                                            
                                            st.rerun()
                                    elif pallet['status'] == 'собрана':
                                        # Отображаем время комплектации
                                        if 'completion_time' in pallet:
                                            st.write(f"Время комплектации: {pallet['completion_time']:.1f} мин.")
                                        
                                        st.success("Комплектация завершена")
                        else:
                            st.info("Нет паллет для комплектации")
            else:
                st.info("Нет активных операторов")
        else:
            st.info("Нет добавленных операторов. Перейдите в раздел Администратор, чтобы добавить операторов.")

    def show_settings_view(self):
        st.header("Настройки системы")
        
        # Настройки паллеты
        with st.expander("Настройки паллеты", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.settings['pallet_extension'] = st.slider(
                    "Вынос паллеты (см)",
                    min_value=1.0,
                    max_value=10.0,
                    value=st.session_state.settings['pallet_extension'],
                    step=0.1
                )
                
                st.session_state.settings['pallet_load_height'] = st.slider(
                    "Высота загрузки паллеты (см)",
                    min_value=50.0,
                    max_value=300.0,
                    value=st.session_state.settings['pallet_load_height'],
                    step=5.0
                )
                
                st.session_state.settings['pallet_weight'] = st.slider(
                    "Вес паллеты (кг)",
                    min_value=100.0,
                    max_value=2000.0,
                    value=st.session_state.settings['pallet_weight'],
                    step=50.0
                )

                st.session_state.settings['pallet_height_diff'] = st.slider(
                    "Допустимый максимальный перепад высот между слоями (см)",
                    min_value=5.0,
                    max_value=50.0,
                    value=st.session_state.settings['pallet_height_diff'],
                    step=1.0
                )
            
            with col2:
                st.session_state.settings['bucket_weight'] = st.slider(
                    "Ставить на ведро (грамм)",
                    min_value=100.0,
                    max_value=1000.0,
                    value=st.session_state.settings['bucket_weight'],
                    step=10.0
                )
                
                # Добавляем новую опцию для заполнения пустот нестандартной формы
                st.session_state.settings['fill_complex_gaps'] = st.checkbox(
                    "Найти и заполнить пустоты нестандартной формы",
                    value=st.session_state.settings.get('fill_complex_gaps', False)
                )
                
                st.session_state.settings['pallet_stability'] = st.slider(
                    "Устойчивость на качение (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=st.session_state.settings['pallet_stability'],
                    step=1.0
                )
        
        # Репорты
        with st.expander("Репорты", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.settings['report_time_at_pallet'] = st.checkbox(
                    "Время нахождения у паллеты более указанного",
                    value=st.session_state.settings['report_time_at_pallet']
                )
                
                if st.session_state.settings['report_time_at_pallet']:
                    st.session_state.settings['report_time_threshold'] = st.slider(
                        "Пороговое время (минут)",
                        min_value=1.0,
                        max_value=20.0,
                        value=st.session_state.settings['report_time_threshold'],
                        step=0.5
                    )
            
            with col2:
                st.session_state.settings['report_operator_pallet'] = st.checkbox(
                    "Есть сформированная оператором паллета",
                    value=st.session_state.settings['report_operator_pallet']
                )
        
        # Кнопка сохранения
        if st.button("Сохранить настройки", key="save_settings"):
            # Здесь можно добавить код для сохранения настроек в файл
            self.add_notification("Настройки успешно сохранены", "success")

    def show_statistics_view(self):
        """Отображение статистики работы операторов и общей статистики склада"""
        st.header("Статистика работы")
        
        # Собираем статистику по всем операторам
        all_completed_pallets = []
        for op_id, pallets in st.session_state.pallets.items():
            completed = [p for p in pallets if p['status'] == 'собрана']
            all_completed_pallets.extend(completed)
        
        if all_completed_pallets:
            # Статистика
            total_pallets = len(all_completed_pallets)
            total_completion_time = sum(p.get('completion_time', 0) for p in all_completed_pallets)
            total_boxes = sum(p.get('box_count', 0) for p in all_completed_pallets)
            avg_time_per_pallet = total_completion_time / total_pallets if total_pallets else 0
            avg_time_per_box = total_completion_time / total_boxes if total_boxes else 0
            
            # Отображаем статистику
            st.write(f"**Всего собрано паллет:** {total_pallets}")
            st.write(f"**Общее время комплектации:** {total_completion_time:.1f} мин.")
            st.write(f"**Всего собрано коробок:** {total_boxes}")
            st.write(f"**Среднее время на паллету:** {avg_time_per_pallet:.1f} мин.")
            st.write(f"**Среднее время на коробку:** {avg_time_per_box:.1f} мин.")
            
            # Статистика по операторам
            operators_stats = []
            for op_id, operator in st.session_state.operators.items():
                if op_id in st.session_state.pallets:
                    completed = [p for p in st.session_state.pallets[op_id] if p['status'] == 'собрана']
                    if completed:
                        op_total_time = sum(p.get('completion_time', 0) for p in completed)
                        op_total_boxes = sum(p.get('box_count', 0) for p in completed)
                        operators_stats.append({
                            "Оператор": operator['name'],
                            "Паллет собрано": len(completed),
                            "Коробок собрано": op_total_boxes,
                            "Общее время (мин)": op_total_time,
                            "Среднее время на паллету (мин)": op_total_time / len(completed) if completed else 0,
                            "Среднее время на коробку (мин)": op_total_time / op_total_boxes if op_total_boxes else 0
                        })
            
            if operators_stats:
                st.subheader("Производительность операторов")
                ops_df = pd.DataFrame(operators_stats)
                st.dataframe(ops_df, use_container_width=True)
                
                # Отображаем график производительности
                st.subheader("График производительности")
                # Создаем столбчатую диаграмму производительности
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[op["Оператор"] for op in operators_stats],
                    y=[op["Коробок собрано"] for op in operators_stats],
                    name="Коробок собрано",
                    marker_color='#2E7D32'  # Зеленый
                ))
                fig.add_trace(go.Bar(
                    x=[op["Оператор"] for op in operators_stats],
                    y=[op["Паллет собрано"] for op in operators_stats],
                    name="Паллет собрано",
                    marker_color='#1976D2'  # Синий
                ))
                fig.update_layout(
                    title="Производительность операторов",
                    xaxis_title="Оператор",
                    yaxis_title="Количество",
                    barmode='group',
                    bargap=0.15,
                    bargroupgap=0.1
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Пока не собрано ни одной паллеты")

    def distribute_orders_to_pallets(self):
        """Распределение артикулов по паллетам операторов с учетом оптимизации пространства"""
        try:
            # Получаем список доступных операторов
            available_operators = [op_id for op_id, op in st.session_state.operators.items() 
                                  if op['status'] == 'готов к работе']
            
            if not available_operators:
                self.add_notification("Нет доступных операторов для распределения артикулов", "warning")
                return
            
            # Проверяем наличие заявок для распределения
            if self.warehouse.order_data is None or len(self.warehouse.order_data) == 0:
                self.add_notification("Нет артикулов для распределения", "warning")
                return
            
            # Максимальное количество итераций распределения
            max_iterations = 7
            iteration_count = 0
            total_pallets_created = 0
            
            # Словарь для хранения количества размещенных коробок по каждому SKU
            boxes_placed = defaultdict(int)
            
            # Проверяем наличие нераспределенных заявок
            while iteration_count < max_iterations:
                # Получаем заявки, которые еще не распределены полностью по паллетам
                available_orders = [
                    index for index, row in self.warehouse.order_data.iterrows()
                    if index not in st.session_state.orders_in_pallets
                ]
                
                # Проверяем, есть ли еще нераспределенные заявки
                if not available_orders:
                    self.add_notification("Все артикулы полностью распределены", "success")
                    break
                
                # Инициализация паллет для операторов, если у них еще нет паллет
                for op_id in available_operators:
                    if op_id not in st.session_state.pallets:
                        st.session_state.pallets[op_id] = []
                
                # Настройки оптимизации паллеты
                optimization_settings = {
                    'center_empty': False,  # Оптимальное использование пространства
                    'prioritize_stability': True,  # Приоритет устойчивости над идеальным заполнением
                    'max_weight': st.session_state.settings['pallet_weight'] * 1000,  # Максимальный вес в граммах
                    'stability_factor': 0.8,  # Минимальное перекрытие для устойчивости (0-1)
                    'max_height': st.session_state.settings['pallet_load_height'],  # Максимальная высота паллеты (2м)
                    'max_overhang': st.session_state.settings['pallet_extension'],  # Максимальный вынос за край паллеты
                    'fill_complex_gaps': st.session_state.settings['fill_complex_gaps'],  # Заполнение пустот нестандартной формы
                    'allow_extra_pallets': True  # Разрешить создание дополнительных паллет для оставшихся коробок
                }
                
                # Подготовка данных для оптимизации
                orders_data = []
                for order_idx in available_orders:
                    order_row = self.warehouse.order_data.iloc[order_idx]
                    
                    # Проверяем, сколько коробок уже распределено для этого SKU
                    sku = order_row.get('SKU')
                    original_quantity = int(order_row.get('Количество', 1))
                    key = f"{order_idx}_{sku}"
                    already_distributed = st.session_state.distributed_boxes.get(key, 0)
                    
                    # Если все коробки этого SKU уже распределены, пропускаем
                    if already_distributed >= original_quantity:
                        continue
                    
                    # Создаем копию данных и обновляем количество оставшихся коробок
                    order_data_copy = order_row.to_dict()
                    remaining_quantity = original_quantity - already_distributed
                    
                    # Если есть оставшиеся коробки, добавляем в заказ
                    if remaining_quantity > 0:
                        order_data_copy['Количество'] = remaining_quantity
                        order_info = {
                            'id': order_idx,
                            'data': order_data_copy
                        }
                        orders_data.append(order_info)
                
                # Если нет заказов для распределения, выходим из цикла
                if not orders_data:
                    break
                
                # Используем алгоритм оптимизации паллет
                optimized_result = optimize_pallet(orders_data, optimization_settings)
                
                # Если оптимизация не удалась, сообщаем об ошибке и прекращаем итерации
                if 'status' in optimized_result and optimized_result['status'] == 'error':
                    self.add_notification(f"Ошибка оптимизации паллеты: {optimized_result.get('message', 'неизвестная ошибка')}", "error")
                    break
                
                # Инициализируем или получаем словарь для хранения сведений о распределенных коробках
                if 'distributed_boxes' not in st.session_state:
                    st.session_state.distributed_boxes = {}
                
                # Распределяем паллеты по операторам
                pallets_created = 0
                total_processed_orders = set()
                
                # Функция для создания новой паллеты из слоев
                def create_pallet_from_layers(pallet_result, op_id):
                    pallet_id = f"pallet_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(st.session_state.pallets[op_id])}"
                    
                    # Извлекаем слои из результата
                    layers = pallet_result.get('layers', [])
                    
                    # Счетчик коробок для каждого заказа (order_id -> {sku -> count})
                    order_box_counts = defaultdict(lambda: defaultdict(int))
                    
                    # Собираем информацию о количестве коробок каждого SKU на паллете
                    total_boxes_in_layers = 0
                    for layer in layers:
                        total_boxes_in_layers += len(layer.get('boxes', []))
                        for box in layer.get('boxes', []):
                            sku = box.get('sku')
                            order_id = box.get('order_id')  # ID заказа
                            
                            if order_id is not None:
                                order_box_counts[order_id][sku] += 1
                    
                    # Собираем все артикулы из слоев
                    orders = []
                    processed_order_ids = set()
                    
                    # Для каждого заказа, имеющегося на этой паллете
                    for order_id, sku_counts in order_box_counts.items():
                        # Если order_id есть в available_orders
                        if order_id in available_orders:
                            order_row = self.warehouse.order_data.iloc[order_id]
                            
                            # Для каждого SKU в заказе
                            for sku, box_count in sku_counts.items():
                                if order_row.get('SKU') == sku:
                                    # Создаем копию данных заказа
                                    order_data = order_row.to_dict()
                                    
                                    # Проверяем, что количество подходит
                                    original_quantity = int(order_row.get('Количество', 1))
                                    
                                    # Получаем уже распределенное количество из предыдущих итераций
                                    key = f"{order_id}_{sku}"
                                    already_distributed = st.session_state.distributed_boxes.get(key, 0)
                                    
                                    # Добавляем информацию о заказе с точным количеством коробок на этой паллете
                                    order_info = {
                                        'id': order_id,
                                        'index': order_id,
                                        'data': order_data,
                                        'box_count': box_count,  # Количество коробок именно на этой паллете
                                        'original_quantity': original_quantity,  # Исходное количество в заказе
                                        'height': next((box.get('height', 0) for layer in layers 
                                                      for box in layer.get('boxes', []) 
                                                      if box.get('sku') == sku), 0),
                                        'placed_in_layers': True  # Отмечаем, что размещено с учетом слоев
                                    }
                                    orders.append(order_info)
                                    
                                    # Отмечаем заказ как обработанный в processed_order_ids для этой паллеты
                                    # Даже если не все коробки размещены, мы отмечаем заказ как частично обработанный
                                    processed_order_ids.add(order_id)
                                    
                                    # Обновляем счетчик распределенных коробок
                                    # Общее распределенное количество = предыдущее + текущее
                                    total_distributed = already_distributed + box_count
                                    
                                    # Только если распределены все коробки, добавляем в глобальный список полностью обработанных заказов
                                    if total_distributed >= original_quantity:
                                        st.session_state.orders_in_pallets.add(order_id)
                                    break
                    
                    # Создаем паллету с информацией о слоях и заказах
                    pallet = {
                        'id': pallet_id,
                        'created_at': datetime.now(),
                        'status': 'новая',
                        'orders': orders,
                        'layers': layers,
                        'total_volume': pallet_result.get('total_volume', 0),
                        'total_weight': pallet_result.get('total_weight', 0),
                        'total_height': sum(layer.get('height', 0) for layer in layers),
                        'box_count': sum(order.get('box_count', 1) for order in orders),
                        'utilization': pallet_result.get('utilization', 0),
                        'stability_score': pallet_result.get('stability_score', 0)
                    }
                    
                    return pallet, processed_order_ids
                
                # Обрабатываем основную паллету
                if 'layers' in optimized_result and optimized_result['layers']:
                    # Выбираем оператора для первой паллеты
                    op_id = available_operators[pallets_created % len(available_operators)]
                    
                    # Создаем паллету
                    pallet, processed = create_pallet_from_layers(optimized_result, op_id)
                    
                    # Добавляем паллету оператору
                    st.session_state.pallets[op_id].append(pallet)
                    
                    # Обновляем счетчики размещенных коробок
                    for order in pallet.get('orders', []):
                        order_id = order.get('id')
                        sku = order.get('data', {}).get('SKU')
                        box_count = order.get('box_count', 0)
                        
                        # Обновляем количество размещенных коробок
                        key = f"{order_id}_{sku}"
                        st.session_state.distributed_boxes[key] = st.session_state.distributed_boxes.get(key, 0) + box_count
                        boxes_placed[key] += box_count
                        
                        # Проверяем, все ли коробки этого SKU распределены
                        total_quantity = int(order.get('original_quantity', 1))
                        
                        # Проверяем, сколько всего коробок этого типа было распределено
                        total_distributed = st.session_state.distributed_boxes[key]
                        
                        # Только если распределены все коробки, добавляем в список обработанных
                        if total_distributed >= total_quantity:
                            total_processed_orders.add(order_id)
                            # Добавляем в глобальный список полностью обработанных заказов
                            st.session_state.orders_in_pallets.add(order_id)
                    
                    # Обновляем счетчики
                    pallets_created += 1
                
                # Обрабатываем дополнительные паллеты
                if 'additional_pallets' in optimized_result:
                    for add_pallet in optimized_result['additional_pallets']:
                        if 'layers' in add_pallet and add_pallet['layers']:
                            # Выбираем следующего оператора циклически
                            op_id = available_operators[pallets_created % len(available_operators)]
                            
                            # Создаем паллету
                            pallet, processed = create_pallet_from_layers(add_pallet, op_id)
                            
                            # Добавляем паллету оператору
                            st.session_state.pallets[op_id].append(pallet)
                            
                            # Обновляем счетчики размещенных коробок
                            for order in pallet.get('orders', []):
                                order_id = order.get('id')
                                sku = order.get('data', {}).get('SKU')
                                box_count = order.get('box_count', 0)
                                
                                # Обновляем количество размещенных коробок
                                key = f"{order_id}_{sku}"
                                st.session_state.distributed_boxes[key] = st.session_state.distributed_boxes.get(key, 0) + box_count
                                boxes_placed[key] += box_count
                                
                                # Проверяем, все ли коробки этого SKU распределены
                                total_quantity = int(order.get('original_quantity', 1))
                                
                                # Проверяем, сколько всего коробок этого типа было распределено
                                total_distributed = st.session_state.distributed_boxes[key]
                                
                                # Только если распределены все коробки, добавляем в список обработанных
                                if total_distributed >= total_quantity:
                                    total_processed_orders.add(order_id)
                                    # Добавляем в глобальный список полностью обработанных заказов
                                    st.session_state.orders_in_pallets.add(order_id)
                            
                            # Обновляем счетчики
                            pallets_created += 1
                
                # Обновляем общий счетчик созданных паллет
                total_pallets_created += pallets_created
                
                # Увеличиваем счетчик итераций
                iteration_count += 1
                
                # Если в этой итерации не удалось создать ни одной паллеты, прекращаем
                if pallets_created == 0:
                    break
                
                # Получаем текущую статистику распределения
                dist_stats = self.get_distribution_stats()
                
                # Если все коробки распределены, выходим из цикла
                if dist_stats['distributed_boxes'] >= dist_stats['total_boxes']:
                    break
            
            # Получаем согласованную статистику после всех итераций
            dist_stats = self.get_distribution_stats()
            
            # Выводим сообщение с информацией о распределении
            if total_pallets_created <= 0:
                self.add_notification("Не удалось распределить артикулы по паллетам", "warning")
            else:
                # Текст уведомления зависит от полноты распределения
                if dist_stats['distributed_boxes'] >= dist_stats['total_boxes']:
                    self.add_notification(
                        f"Успешно распределены все {dist_stats['distributed_boxes']} коробок "
                        f"({dist_stats['distribution_percent']:.1f}%). "
                        f"Обработано {dist_stats['distributed_orders']} из {dist_stats['total_orders']} артикулов, "
                        f"создано {total_pallets_created} паллет за {iteration_count} итераций.", 
                        "success"
                    )
                else:
                    self.add_notification(
                        f"Распределено {dist_stats['distributed_boxes']} из {dist_stats['total_boxes']} коробок "
                        f"({dist_stats['distribution_percent']:.1f}%). "
                        f"Осталось распределить {dist_stats['total_boxes'] - dist_stats['distributed_boxes']} коробок. "
                        f"Обработано {dist_stats['distributed_orders']} из {dist_stats['total_orders']} артикулов, "
                        f"создано {total_pallets_created} паллет за {iteration_count} итераций.",
                        "warning" if dist_stats['distributed_boxes'] < dist_stats['total_boxes'] else "success"
                    )
            
        except Exception as e:
            self.add_notification(f"Ошибка при распределении артикулов: {str(e)}", "error")
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    st.set_page_config(
        page_title="Оптимизация процесса паллетизации товаров",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    app = WarehouseApp()
    app.run()
