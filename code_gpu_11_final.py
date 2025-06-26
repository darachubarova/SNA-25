import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import spacy
import pickle
import os
import re
from collections import defaultdict
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import warnings

import matplotlib
matplotlib.use('Agg')  # Для работы без дисплея
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.patches as mpatches
from collections import Counter

warnings.filterwarnings('ignore')

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

def calculate_metrics(y_true, y_pred):
    """Вычисление Precision, Recall, F1"""
    try:
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return precision, recall, f1
    except:
        return 0.0, 0.0, 0.0

class FeverDataProcessor:
    """Класс для обработки данных FEVER"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Модель en_core_web_sm не найдена. Устанавливаем...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')
        
    def load_and_preprocess_data(self, max_samples=15000):
        """Загрузка и предварительная обработка данных FEVER"""
        print("Этап 1: Загрузка и предварительная обработка данных FEVER")
        
        data = []
        label_mapping = {
            'SUPPORTS': 0,      # Не противоречие
            'REFUTES': 1,       # Противоречие
            'NOT ENOUGH INFO': 2 # Нейтральное
        }
        
        sample_count = 0
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Загрузка данных", total=max_samples)):
                if i >= max_samples:
                    break
                    
                try:
                    item = json.loads(line.strip())
                    
                    if i < 3:
                        print(f"\nОтладка записи {i}:")
                        print(f"Ключи: {item.keys()}")
                        if 'evidence' in item:
                            print(f"Evidence структура: {type(item['evidence'])}")
                            print(f"Evidence содержимое: {item['evidence']}")
                    
                    if item['label'] in label_mapping:
                        evidence_text = self._extract_evidence_text(item.get('evidence', []))
                        
                        processed_item = {
                            'id': item.get('id', i),
                            'claim': item['claim'],
                            'evidence': evidence_text,
                            'label': label_mapping[item['label']],
                            'label_name': item['label']
                        }
                        data.append(processed_item)
                        sample_count += 1
                        
                except json.JSONDecodeError as e:
                    print(f"Ошибка парсинга JSON в строке {i}: {e}")
                    continue
                except Exception as e:
                    print(f"Ошибка обработки записи {i}: {e}")
                    continue
        
        print(f"Загружено {len(data)} образцов")
        
        if data:
            print("\nПример загруженных данных:")
            for i in range(min(2, len(data))):
                print(f"Пример {i+1}:")
                print(f"  Claim: {data[i]['claim'][:100]}...")
                print(f"  Evidence: {data[i]['evidence'][:100]}...")
                print(f"  Label: {data[i]['label_name']}")
                
        return data
    
    def _extract_evidence_text(self, evidence):
        """Безопасное извлечение текста из evidence"""
        if not evidence:
            return ""
        
        evidence_texts = []
        
        try:
            if isinstance(evidence, list):
                for ev_group in evidence:
                    if isinstance(ev_group, list):
                        for ev_item in ev_group:
                            if isinstance(ev_item, list) and len(ev_item) >= 3:
                                evidence_texts.append(str(ev_item[2]))
                            elif isinstance(ev_item, str):
                                evidence_texts.append(ev_item)
                            elif isinstance(ev_item, dict) and 'text' in ev_item:
                                evidence_texts.append(ev_item['text'])
                    elif isinstance(ev_group, str):
                        evidence_texts.append(ev_group)
                    elif isinstance(ev_group, dict) and 'text' in ev_group:
                        evidence_texts.append(ev_group['text'])
            elif isinstance(evidence, str):
                evidence_texts.append(evidence)
                
        except Exception as e:
            print(f"⚠️ Ошибка извлечения evidence: {e}")
            return ""
        
        return " ".join(evidence_texts) if evidence_texts else ""

class EntityExtractor:
    """Класс для извлечения сущностей и построения KG"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("⚠️ Модель en_core_web_sm не найдена. Устанавливаем...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')
            
        self.knowledge_graph = nx.DiGraph()
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.relation_to_id = {}
        self.id_to_relation = {}
        
    def extract_entities_and_relations(self, data):
        """Извлечение сущностей и отношений из текста"""
        print("Этап 2: Извлечение сущностей и построение графа знаний")
        
        all_entities = set()
        all_relations = []
        entity_pairs = []
        
        processed_count = 0
        
        for item in tqdm(data, desc="Извлечение сущностей"):
            try:
                claim_entities = self._extract_entities_from_text(item['claim'])
                evidence_entities = self._extract_entities_from_text(item['evidence'])
                
                # Фильтрация сущностей по частоте и релевантности
                claim_entities = self._filter_entities(claim_entities)
                evidence_entities = self._filter_entities(evidence_entities)
                
                all_entities.update(claim_entities)
                all_entities.update(evidence_entities)
                
                for claim_ent in claim_entities:
                    for ev_ent in evidence_entities:
                        if claim_ent != ev_ent:
                            relation = self._determine_relation(item['label'])
                            all_relations.append((claim_ent, relation, ev_ent))
                            entity_pairs.append((claim_ent, ev_ent, relation, item['label']))
                
                processed_count += 1
                
            except Exception as e:
                print(f"Ошибка обработки сущностей для записи: {e}")
                continue
        
        print(f"Обработано {processed_count} записей")
        print(f"Найдено {len(all_entities)} уникальных сущностей")
        
        if not all_entities:
            print("Не найдено сущностей. Создаем базовые сущности...")
            for item in data[:100]:
                words = item['claim'].split()[:5]
                for word in words:
                    if len(word) > 3:
                        all_entities.add(word.lower())
        
        self.entity_to_id = {ent: i for i, ent in enumerate(all_entities)}
        self.id_to_entity = {i: ent for ent, i in self.entity_to_id.items()}
        
        unique_relations = list(set([rel[1] for rel in all_relations])) if all_relations else ['supports', 'contradicts', 'neutral']
        self.relation_to_id = {rel: i for i, rel in enumerate(unique_relations)}
        self.id_to_relation = {i: rel for rel, i in self.relation_to_id.items()}
        
        edges_added = 0
        for head, relation, tail in tqdm(all_relations, desc="Построение KG"):
            try:
                self.knowledge_graph.add_edge(
                    self.entity_to_id[head],
                    self.entity_to_id[tail],
                    relation=self.relation_to_id[relation]
                )
                edges_added += 1
            except Exception as e:
                continue
        
        print(f"Создан граф с {len(all_entities)} сущностями и {edges_added} отношениями")

        # Создание визуализаций графа знаний
        try:
            visualize_knowledge_graph(self, 'charts/knowledge_graph.png')
            plot_kg_statistics(self, 'charts/kg_statistics.png')
        except Exception as e:
            print(f"⚠️ Ошибка создания визуализации: {e}")

        return entity_pairs
    
    def _extract_entities_from_text(self, text):
        """Извлечение именованных сущностей из текста"""
        if not text or not isinstance(text, str):
            return []
            
        try:
            doc = self.nlp(text[:15000])
            entities = []
            
            # Ограничение типов сущностей для повышения релевантности
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']:
                    clean_text = ent.text.lower().strip()
                    if len(clean_text) > 2 and not re.match(r'^\d+$', clean_text):
                        entities.append(clean_text)
            
            # Добавление только имен собственных и ключевых существительных
            for chunk in doc.noun_chunks:
                if chunk.root.pos_ in ['NOUN', 'PROPN'] and len(chunk.text) > 2:
                    clean_text = chunk.root.lemma_.lower()
                    if clean_text.isalpha() and len(clean_text) > 3:
                        entities.append(clean_text)
                        
            return list(set(entities))
            
        except Exception as e:
            print(f"⚠️ Ошибка NLP обработки: {e}")
            words = text.split()
            return [word.lower() for word in words if len(word) > 3 and word.isalpha()]
    
    def _filter_entities(self, entities):
        """Фильтрация сущностей по релевантности"""
        if not entities:
            return []
        # Удаление слишком общих или неинформативных сущностей
        stop_entities = {'thing', 'something', 'nothing', 'anything'}
        filtered = [e for e in entities if e not in stop_entities and len(e.split()) <= 3]
        return filtered[:10]  # Ограничение числа сущностей для снижения шума
    
    def _determine_relation(self, label):
        """Определение типа отношения на основе метки"""
        if label == 0:  # SUPPORTS
            return 'supports'
        elif label == 1:  # REFUTES
            return 'contradicts'
        else:  # NOT ENOUGH INFO
            return 'neutral'

class KGEmbedding(nn.Module):
    """Модель для эмбеддингов графа знаний (DistMult)"""
    
    def __init__(self, num_entities, num_relations, embed_dim=200):
        super(KGEmbedding, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        
        self.entity_embeddings = nn.Embedding(num_entities, embed_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # Нормализация эмбеддингов
        self.normalize_entities()
        
    def normalize_entities(self):
        """Нормализация эмбеддингов сущностей"""
        with torch.no_grad():
            self.entity_embeddings.weight.div_(
                torch.norm(self.entity_embeddings.weight, dim=1, keepdim=True).clamp(min=1e-12)
            )

    def forward(self, heads, relations, tails):
        """Вычисление скора для триплетов (TransE вместо DistMult)"""
        head_embeds = self.entity_embeddings(heads)
        relation_embeds = self.relation_embeddings(relations)
        tail_embeds = self.entity_embeddings(tails)
    
        # TransE: h + r ≈ t, score = -||h + r - t||
        scores = head_embeds + relation_embeds - tail_embeds
        distances = torch.norm(scores, p=2, dim=1)
        return distances  # Возвращаем расстояния (меньше = лучше)

class ContradictionDataset(Dataset):
    """Dataset для обучения модели обнаружения противоречий"""
    
    def __init__(self, data, tokenizer, entity_extractor, kg_model, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.entity_extractor = entity_extractor
        self.kg_model = kg_model
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Комбинирование claim и evidence
        claim = str(item.get('claim', ''))
        evidence = str(item.get('evidence', ''))
        text = f"{claim} [SEP] {evidence}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Извлечение сущностей
        claim_entities = self.entity_extractor._extract_entities_from_text(claim)
        evidence_entities = self.entity_extractor._extract_entities_from_text(evidence)
        
        # Получение эмбеддингов сущностей
        entity_ids = []
        for entity in set(claim_entities + evidence_entities):
            if entity in self.entity_extractor.entity_to_id:
                entity_ids.append(self.entity_extractor.entity_to_id[entity])
        
        # Агрегация эмбеддингов (взвешенное усреднение)
        if entity_ids:
            entity_ids_tensor = torch.tensor(entity_ids, dtype=torch.long).to(device)
            with torch.no_grad():
                entity_embeds = self.kg_model.entity_embeddings(entity_ids_tensor)
                # Взвешивание по обратной частоте сущностей
                weights = torch.ones(len(entity_ids), device=device) / (len(entity_ids) + 1e-6)
                kg_embedding = torch.sum(entity_embeds * weights.unsqueeze(1), dim=0)
        else:
            kg_embedding = torch.zeros(self.kg_model.embed_dim).to(device)
            
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(item['label'], dtype=torch.long),
            'kg_embedding': kg_embedding
        }

class ContradictionClassifier(nn.Module):
    """Модель классификации противоречий с кросс-аттеншеном"""
    
    def __init__(self, pretrained_model_name, num_classes=3, kg_embed_dim=200):
        super(ContradictionClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)
        
        bert_dim = self.bert.config.hidden_size
        self.kg_projection = nn.Linear(kg_embed_dim, bert_dim)
        
        # Кросс-аттеншен слой
        self.cross_attention = nn.MultiheadAttention(bert_dim, num_heads=8)
        
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, kg_embeddings=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        if kg_embeddings is not None:
            # Проецируем KG эмбеддинги в пространство BERT
            kg_projected = self.kg_projection(kg_embeddings)  # [batch_size, hidden_size]
            kg_projected = kg_projected.unsqueeze(1)  # [batch_size, 1, hidden_size]
            
            # Кросс-аттеншен между sequence_output и kg_projected
            sequence_output = sequence_output.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
            attn_output, _ = self.cross_attention(
                kg_projected.transpose(0, 1),  # query: [1, batch_size, hidden_size]
                sequence_output,               # key: [seq_len, batch_size, hidden_size]
                sequence_output               # value: [seq_len, batch_size, hidden_size]
            )
            attn_output = attn_output.transpose(0, 1).squeeze(1)  # [batch_size, hidden_size]
            
            combined = self.dropout(attn_output)
        else:
            combined = self.dropout(pooled_output)
            
        logits = self.classifier(combined)
        return logits

class SimpleContradictionClassifier(nn.Module):
    """Модель классификации противоречий без KG"""
    
    def __init__(self, pretrained_model_name, num_classes=3):
        super(SimpleContradictionClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)
        
        bert_dim = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def visualize_knowledge_graph(entity_extractor, save_path='charts/kg_visualization.png', max_nodes=100):
    """Визуализация графа знаний"""
    print("🎨 Создание визуализации графа знаний...")
    
    plt.figure(figsize=(15, 12))
    
    # Ограничиваем количество узлов для читаемости
    G = entity_extractor.knowledge_graph
    if len(G.nodes()) > max_nodes:
        # Берем узлы с наибольшим количеством связей
        node_degrees = dict(G.degree())
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_ids = [node[0] for node in top_nodes]
        G = G.subgraph(top_node_ids)
    
    # Создание layout
    if len(G.nodes()) > 0:
        try:
            pos = nx.spring_layout(G, k=3, iterations=50)
        except:
            pos = nx.random_layout(G)
        
        # Определение цветов для разных типов отношений
        relation_colors = {0: 'green', 1: 'red', 2: 'gray'}  # supports, contradicts, neutral
        
        # Рисование рёбер с разными цветами
        for edge in G.edges(data=True):
            relation = edge[2].get('relation', 2)
            color = relation_colors.get(relation, 'gray')
            nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])],
                                 edge_color=color, alpha=0.6, width=1.5)
        
        # Рисование узлов
        node_sizes = [300 + G.degree(node) * 50 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                             node_size=node_sizes, alpha=0.8)
        
        # Добавление подписей для важных узлов
        important_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)#[:20]
        labels = {}
        for node_id, _ in important_nodes:
            if node_id in entity_extractor.id_to_entity:
                entity_name = entity_extractor.id_to_entity[node_id]
                labels[node_id] = entity_name[:15] + "..." if len(entity_name) > 15 else entity_name
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Легенда
        legend_elements = [
            mpatches.Patch(color='green', label='Supports'),
            mpatches.Patch(color='red', label='Contradicts'),
            mpatches.Patch(color='gray', label='Neutral')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f'Knowledge Graph Visualization\n'
                 f'Nodes: {len(G.nodes())}, Edges: {len(G.edges())}',
                 fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No graph data available',
                ha='center', va='center', fontsize=16)
        plt.title('Knowledge Graph Visualization', fontsize=14)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Граф сохранён: {save_path}")

def plot_training_curves(train_losses_kg, val_losses_kg, train_accs_kg, val_accs_kg,
                        train_losses_no_kg, val_losses_no_kg, train_accs_no_kg, val_accs_no_kg,
                        train_precision_kg, val_precision_kg, train_recall_kg, val_recall_kg, train_f1_kg, val_f1_kg,
                        train_precision_no_kg, val_precision_no_kg, train_recall_no_kg, val_recall_no_kg, train_f1_no_kg, val_f1_no_kg,
                        save_path='charts/training_curves_detailed.png'):
    """Построение детальных графиков обучения для обеих моделей"""
    print("📊 Создание детальных графиков обучения...")
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    epochs = range(1, len(train_losses_kg) + 1)
    
    # График потерь для модели с KG
    axes[0, 0].plot(epochs, train_losses_kg, 'b-', label='Training Loss (KG)', linewidth=2, marker='o')
    axes[0, 0].plot(epochs, val_losses_kg, 'r-', label='Validation Loss (KG)', linewidth=2, marker='s')
    axes[0, 0].set_title('Model with Knowledge Graph - Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # График потерь для модели без KG
    axes[0, 1].plot(epochs, train_losses_no_kg, 'b--', label='Training Loss (No KG)', linewidth=2, marker='o')
    axes[0, 1].plot(epochs, val_losses_no_kg, 'r--', label='Validation Loss (No KG)', linewidth=2, marker='s')
    axes[0, 1].set_title('Model without Knowledge Graph - Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # График точности для обеих моделей (сравнение)
    axes[1, 0].plot(epochs, train_accs_kg, 'b-', label='Training Accuracy (KG)', linewidth=2, marker='o')
    axes[1, 0].plot(epochs, val_accs_kg, 'r-', label='Validation Accuracy (KG)', linewidth=2, marker='s')
    axes[1, 0].plot(epochs, train_accs_no_kg, 'b--', label='Training Accuracy (No KG)', linewidth=2, marker='^')
    axes[1, 0].plot(epochs, val_accs_no_kg, 'r--', label='Validation Accuracy (No KG)', linewidth=2, marker='d')
    axes[1, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # График Precision для обеих моделей
    axes[1, 1].plot(epochs, train_precision_kg, 'g-', label='Training Precision (KG)', linewidth=2, marker='o')
    axes[1, 1].plot(epochs, val_precision_kg, 'orange', label='Validation Precision (KG)', linewidth=2, marker='s')
    axes[1, 1].plot(epochs, train_precision_no_kg, 'g--', label='Training Precision (No KG)', linewidth=2, marker='^')
    axes[1, 1].plot(epochs, val_precision_no_kg, color='orange', linestyle='--', label='Validation Precision (No KG)', linewidth=2, marker='d')
    axes[1, 1].set_title('Precision Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # График Recall для обеих моделей
    axes[2, 0].plot(epochs, train_recall_kg, 'purple', label='Training Recall (KG)', linewidth=2, marker='o')
    axes[2, 0].plot(epochs, val_recall_kg, 'brown', label='Validation Recall (KG)', linewidth=2, marker='s')
    axes[2, 0].plot(epochs, train_recall_no_kg, 'purple', linestyle='--', label='Training Recall (No KG)', linewidth=2, marker='^')
    axes[2, 0].plot(epochs, val_recall_no_kg, 'brown', linestyle='--', label='Validation Recall (No KG)', linewidth=2, marker='d')
    axes[2, 0].set_title('Recall Comparison', fontsize=14, fontweight='bold')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Recall')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # График F1-Score для обеих моделей
    axes[2, 1].plot(epochs, train_f1_kg, 'teal', label='Training F1 (KG)', linewidth=2, marker='o')
    axes[2, 1].plot(epochs, val_f1_kg, 'navy', label='Validation F1 (KG)', linewidth=2, marker='s')
    axes[2, 1].plot(epochs, train_f1_no_kg, 'teal', linestyle='--', label='Training F1 (No KG)', linewidth=2, marker='^')
    axes[2, 1].plot(epochs, val_f1_no_kg, 'navy', linestyle='--', label='Validation F1 (No KG)', linewidth=2, marker='d')
    axes[2, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('F1-Score')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Детальные графики обучения сохранены: {save_path}")
    pd.DataFrame([train_losses_kg, val_losses_kg, train_accs_kg, val_accs_kg,
                        train_losses_no_kg, val_losses_no_kg, train_accs_no_kg, val_accs_no_kg,
                        train_precision_kg, val_precision_kg, train_recall_kg, val_recall_kg, train_f1_kg, val_f1_kg,
                        train_precision_no_kg, val_precision_no_kg, train_recall_no_kg, val_recall_no_kg, train_f1_no_kg, val_f1_no_kg]).to_csv('charts/res.csv')
    print(f"✅ Статистики обучения сохранены: charts/res.csv")
    

def plot_metrics_summary(metrics_kg, metrics_no_kg, save_path='charts/metrics_summary.png'):
    """Сводный график сравнения финальных метрик"""
    print("📈 Создание сводного графика метрик...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Барная диаграмма сравнения метрик
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, metrics_kg, width, label='With KG', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, metrics_no_kg, width, label='Without KG', color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Добавление значений на барах
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # Радарная диаграмма
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]  # Замыкание круга
    
    metrics_kg_radar = metrics_kg + [metrics_kg[0]]  # Замыкание для радара
    metrics_no_kg_radar = metrics_no_kg + [metrics_no_kg[0]]
    
    ax2 = plt.subplot(122, projection='polar')
    ax2.plot(angles, metrics_kg_radar, 'o-', linewidth=2, label='With KG', color='blue')
    ax2.fill(angles, metrics_kg_radar, alpha=0.25, color='blue')
    ax2.plot(angles, metrics_no_kg_radar, 'o-', linewidth=2, label='Without KG', color='red')
    ax2.fill(angles, metrics_no_kg_radar, alpha=0.25, color='red')
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics_names)
    ax2.set_ylim(0, 1)
    ax2.set_title('Performance Radar Chart', fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Сводный график метрик сохранён: {save_path}")

def plot_confusion_matrices(y_true, y_pred_kg, y_pred_no_kg, class_names, save_path='charts/confusion_matrices.png'):
    """Построение матриц ошибок для обеих моделей"""
    print("🎯 Создание матриц ошибок...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Матрица ошибок для модели с KG
    cm_kg = confusion_matrix(y_true, y_pred_kg)
    sns.heatmap(cm_kg, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax1)
    
    accuracy_kg = np.trace(cm_kg) / np.sum(cm_kg)
    ax1.set_title(f'Model with KG\nAccuracy: {accuracy_kg:.3f}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    
    # Матрица ошибок для модели без KG
    cm_no_kg = confusion_matrix(y_true, y_pred_no_kg)
    sns.heatmap(cm_no_kg, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax2)
    
    accuracy_no_kg = np.trace(cm_no_kg) / np.sum(cm_no_kg)
    ax2.set_title(f'Model without KG\nAccuracy: {accuracy_no_kg:.3f}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Матрицы ошибок сохранены: {save_path}")

def plot_kg_statistics(entity_extractor, save_path='charts/kg_statistics.png'):
    """Статистика графа знаний"""
    print("📈 Создание статистики графа знаний...")
    
    G = entity_extractor.knowledge_graph
    
    if len(G.nodes()) == 0:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No graph data available', ha='center', va='center', fontsize=16)
        plt.title('Knowledge Graph Statistics')
        plt.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Распределение степеней узлов
    degrees = [G.degree(n) for n in G.nodes()]
    ax1.hist(degrees, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Node Degree Distribution')

class ContradictionDetectionPipeline:
    """Основной пайплайн для обнаружения противоречий"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.device = device
        self.data_processor = FeverDataProcessor(data_path)
        self.entity_extractor = EntityExtractor()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
    def run_full_pipeline(self, max_samples=5000):
        """Запуск полного пайплайна"""
        print("Запуск пайплайна для обнаружения противоречий\n")
        
        # Этап 1: Загрузка и обработка данных
        data = self.data_processor.load_and_preprocess_data(max_samples)
        
        if len(data) == 0:
            print("Не удалось загрузить данные!")
            return None, None
        
        # Этап 2: Извлечение сущностей и построение KG
        entity_pairs = self.entity_extractor.extract_entities_and_relations(data)
        
        # Этап 3: Обучение KG эмбеддингов
        kg_model = self.train_kg_embeddings()
        
        # Этап 4: Подготовка данных для обучения
        print("Этап 4: Подготовка данных для обучения")
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, stratify=[item['label'] for item in data])
        
        train_dataset = ContradictionDataset(train_data, self.tokenizer, self.entity_extractor, kg_model)
        val_dataset = ContradictionDataset(val_data, self.tokenizer, self.entity_extractor, kg_model)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        print(f"Подготовлены данные: {len(train_data)} для обучения, {len(val_data)} для валидации")
        
        # Этап 5: Обучение моделей
        model_with_kg, model_without_kg = self.train_classification_model(train_loader, val_loader, kg_model)
        
        # Этап 6: Оценка моделей
        accuracy_kg, accuracy_no_kg, report = self.evaluate_model(model_with_kg, model_without_kg, val_loader)
        
        return model_with_kg, model_without_kg
    
    def train_kg_embeddings(self):
        """Обучение эмбеддингов графа знаний"""
        print("Этап 3: Обучение эмбеддингов графа знаний")
    
        num_entities = len(self.entity_extractor.entity_to_id)
        num_relations = len(self.entity_extractor.relation_to_id)
    
        if num_entities == 0 or num_relations == 0:
            print("Недостаточно сущностей или отношений для обучения KG")
            kg_model = KGEmbedding(max(1, num_entities), max(1, num_relations)).to(self.device)
            return kg_model
        
        kg_model = KGEmbedding(num_entities, num_relations).to(self.device)
        optimizer = optim.Adam(kg_model.parameters(), lr=0.001, weight_decay=1e-5)  # Меньше LR
    
        edges = list(self.entity_extractor.knowledge_graph.edges(data=True))
    
        if len(edges) == 0:
            print("Граф знаний пуст")
            return kg_model
        
        kg_model.train()
        num_epochs = 20
    
        for epoch in tqdm(range(num_epochs), desc="Обучение KG эмбеддингов"):
            total_loss = 0
            batch_size = 256  # Уменьшили batch size
            batch_count = 0
            valid_batches = 0
        
            for i in range(0, len(edges), batch_size):
                batch_edges = edges[i:i+batch_size]
            
                if len(batch_edges) == 0:
                    continue
                
                try:
                    heads = torch.tensor([edge[0] for edge in batch_edges], dtype=torch.long).to(self.device)
                    tails = torch.tensor([edge[1] for edge in batch_edges], dtype=torch.long).to(self.device)
                    relations = torch.tensor([edge[2]['relation'] for edge in batch_edges], dtype=torch.long).to(self.device)
                
                    optimizer.zero_grad()
                
                    # Положительные примеры
                    pos_scores = kg_model(heads, relations, tails)
                
                    # Отрицательные примеры
                    neg_tails = torch.randint(0, num_entities, tails.shape).to(self.device)
                    neg_heads = torch.randint(0, num_entities, heads.shape).to(self.device)
                
                    neg_scores_tails = kg_model(heads, relations, neg_tails, negative=True)
                    neg_scores_heads = kg_model(neg_heads, relations, tails, negative=True)
                
                    # Стабильная функция потерь
                    pos_scores = torch.clamp(pos_scores, 1e-7, 1-1e-7)
                    neg_scores_tails = torch.clamp(neg_scores_tails, 1e-7, 1-1e-7)
                    neg_scores_heads = torch.clamp(neg_scores_heads, 1e-7, 1-1e-7)
                
                    loss = (-torch.mean(torch.log(pos_scores))
                        - 0.5 * torch.mean(torch.log(1 - neg_scores_tails))
                        - 0.5 * torch.mean(torch.log(1 - neg_scores_heads)))
                
                    # Проверка на NaN
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️ Некорректная потеря в эпохе {epoch}, пропускаем батч")
                        continue
                    
                    loss.backward()
                
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(kg_model.parameters(), max_norm=1.0)
                
                    optimizer.step()
                    kg_model.normalize_entities()
                
                    total_loss += loss.item()
                    valid_batches += 1
                
                except Exception as e:
                    print(f"⚠️ Ошибка в батче: {e}")
                    continue
                
                batch_count += 1
        
            if valid_batches > 0:
                avg_loss = total_loss / valid_batches
                if epoch % 10 == 0:
                    print(f"Эпоха {epoch}, средняя потеря: {avg_loss:.4f}, валидных батчей: {valid_batches}/{batch_count}")
            else:
                print(f"⚠️ Эпоха {epoch}: нет валидных батчей")
    
        print("✅ Обучение KG завершено")
        return kg_model
    
    def train_classification_model(self, train_loader, val_loader, kg_model):
        """Обучение моделей классификации (с KG и без KG)"""
        print("Этап 5: Обучение моделей классификации противоречий")
        
        model_with_kg = ContradictionClassifier('bert-base-uncased').to(self.device)
        model_without_kg = SimpleContradictionClassifier('bert-base-uncased').to(self.device)
        
        optimizer_kg = optim.AdamW(model_with_kg.parameters(), lr=2e-5)
        optimizer_no_kg = optim.AdamW(model_without_kg.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        
        # ДОБАВИТЬ ЭТИ ПЕРЕМЕННЫЕ ДЛЯ ГРАФИКОВ:
        train_losses_kg = []
        val_losses_kg = []
        train_accs_kg = []
        val_accs_kg = []
        train_losses_no_kg = []
        val_losses_no_kg = []
        train_accs_no_kg = []
        val_accs_no_kg = []
        
        # Для графиков обучения (добавить к существующим переменным)
        train_precision_kg = []
        val_precision_kg = []
        train_recall_kg = []
        val_recall_kg = []
        train_f1_kg = []
        val_f1_kg = []

        train_precision_no_kg = []
        val_precision_no_kg = []
        train_recall_no_kg = []
        val_recall_no_kg = []
        train_f1_no_kg = []
        val_f1_no_kg = []
        
        num_epochs = 2
        best_val_acc_kg = 0
        best_val_acc_no_kg = 0
        
        for epoch in range(num_epochs):
            print(f"\nЭпоха {epoch + 1}/{num_epochs}")
            
            # Обучение модели с KG
            model_with_kg.train()
            train_loss_kg = 0
            train_correct_kg = 0
            train_total_kg = 0
            
            train_pbar = tqdm(train_loader, desc=f"Обучение (KG) эпоха {epoch + 1}")
            # В цикле обучения KG модели, добавить:
            epoch_train_losses_kg = []
            epoch_train_preds_kg = []
            epoch_train_labels_kg = []
            for batch_idx, batch in enumerate(train_pbar):
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    kg_embeddings = batch['kg_embedding'].to(self.device)
                    
                    optimizer_kg.zero_grad()
                    
                    outputs = model_with_kg(input_ids, attention_mask, kg_embeddings)
                    loss = criterion(outputs, labels)
                    epoch_train_losses_kg.append(loss.item())  # ← ДОБАВИТЬ
                    loss.backward()
                    optimizer_kg.step()
                    
                    train_loss_kg += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    epoch_train_preds_kg.extend(predicted.cpu().numpy())
                    epoch_train_labels_kg.extend(labels.cpu().numpy())
                    train_total_kg += labels.size(0)
                    train_correct_kg += (predicted == labels).sum().item()
                    
                    train_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100 * train_correct_kg / train_total_kg:.2f}%'
                    })
                    
                except Exception as e:
                    print(f"Ошибка в батче (KG) {batch_idx}: {e}")
                    continue
            
            # Обучение модели без KG
            model_without_kg.train()
            train_loss_no_kg = 0
            train_correct_no_kg = 0
            train_total_no_kg = 0
            
            train_pbar_no_kg = tqdm(train_loader, desc=f"Обучение (без KG) эпоха {epoch + 1}")
            # В цикле обучения модели без KG, добавить:
            epoch_train_preds_no_kg = []
            epoch_train_labels_no_kg = []
            epoch_train_losses_no_kg = []
            for batch_idx, batch in enumerate(train_pbar_no_kg):
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    optimizer_no_kg.zero_grad()
                    
                    outputs = model_without_kg(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    epoch_train_losses_no_kg.append(loss.item())  # ← ДОБАВИТЬ
                    
                    loss.backward()
                    optimizer_no_kg.step()
                    
                    train_loss_no_kg += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    epoch_train_preds_no_kg.extend(predicted.cpu().numpy())
                    epoch_train_labels_no_kg.extend(labels.cpu().numpy())
                    train_total_no_kg += labels.size(0)
                    train_correct_no_kg += (predicted == labels).sum().item()
                    
                    train_pbar_no_kg.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100 * train_correct_no_kg / train_total_no_kg:.2f}%'
                    })
                    
                except Exception as e:
                    print(f"Ошибка в батче (без KG) {batch_idx}: {e}")
                    continue
            
            # Валидация
            model_with_kg.eval()
            model_without_kg.eval()
            val_loss_kg = 0
            val_correct_kg = 0
            val_total_kg = 0
            val_loss_no_kg = 0
            val_correct_no_kg = 0
            val_total_no_kg = 0
            
            # В цикле валидации добавить:
            epoch_val_losses_kg = []
            epoch_val_losses_no_kg = []
            epoch_val_preds_kg = []
            epoch_val_labels_kg = []
            epoch_val_preds_no_kg = []
            epoch_val_labels_no_kg = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Валидация эпоха {epoch + 1}")
                for batch in val_pbar:
                    try:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['label'].to(self.device)
                        kg_embeddings = batch['kg_embedding'].to(self.device)
                        
                        outputs_kg = model_with_kg(input_ids, attention_mask, kg_embeddings)
                        loss_kg = criterion(outputs_kg, labels)
                        epoch_val_losses_kg.append(loss_kg.item())  # ← ДОБАВИТЬ
                        
                        val_loss_kg += loss_kg.item()
                        _, predicted_kg = torch.max(outputs_kg.data, 1)
                        val_total_kg += labels.size(0)
                        val_correct_kg += (predicted_kg == labels).sum().item()
                        
                        outputs_no_kg = model_without_kg(input_ids, attention_mask)
                        loss_no_kg = criterion(outputs_no_kg, labels)
                        epoch_val_losses_no_kg.append(loss_no_kg.item())  # ← ДОБАВИТЬ
                        
                        val_loss_no_kg += loss_no_kg.item()
                        _, predicted_no_kg = torch.max(outputs_no_kg.data, 1)
                        val_total_no_kg += labels.size(0)
                        val_correct_no_kg += (predicted_no_kg == labels).sum().item()
                        
                        epoch_val_preds_kg.extend(predicted_kg.cpu().numpy())
                        epoch_val_labels_kg.extend(labels.cpu().numpy())
                        epoch_val_preds_no_kg.extend(predicted_no_kg.cpu().numpy())
                        epoch_val_labels_no_kg.extend(labels.cpu().numpy())
                        
                    except Exception as e:
                        continue
                        
                    val_pbar.set_postfix({
                        'Loss (KG)': f'{loss_kg.item():.4f}',
                        'Acc (KG)': f'{100 * val_correct_kg / val_total_kg:.2f}%',
                        'Loss (No KG)': f'{loss_no_kg.item():.4f}',
                        'Acc (No KG)': f'{100 * val_correct_no_kg / val_total_no_kg:.2f}%'
                    })
            
            
            val_acc_kg = 100 * val_correct_kg / val_total_kg if val_total_kg > 0 else 0
            val_acc_no_kg = 100 * val_correct_no_kg / val_total_no_kg if val_total_no_kg > 0 else 0
            
            # После вычисления val_acc_kg и val_acc_no_kg добавить:

            # Вычисление метрик для модели с KG
            train_prec_kg, train_rec_kg, train_f1_kg_val = calculate_metrics(epoch_train_labels_kg, epoch_train_preds_kg)
            val_prec_kg, val_rec_kg, val_f1_kg_val = calculate_metrics(epoch_val_labels_kg, epoch_val_preds_kg)

            train_precision_kg.append(train_prec_kg)
            train_recall_kg.append(train_rec_kg)
            train_f1_kg.append(train_f1_kg_val)
            val_precision_kg.append(val_prec_kg)
            val_recall_kg.append(val_rec_kg)
            val_f1_kg.append(val_f1_kg_val)

            # Вычисление метрик для модели без KG
            train_prec_no_kg, train_rec_no_kg, train_f1_no_kg_val = calculate_metrics(epoch_train_labels_no_kg, epoch_train_preds_no_kg)
            val_prec_no_kg, val_rec_no_kg, val_f1_no_kg_val = calculate_metrics(epoch_val_labels_no_kg, epoch_val_preds_no_kg)

            train_precision_no_kg.append(train_prec_no_kg)
            train_recall_no_kg.append(train_rec_no_kg)
            train_f1_no_kg.append(train_f1_no_kg_val)
            val_precision_no_kg.append(val_prec_no_kg)
            val_recall_no_kg.append(val_rec_no_kg)
            val_f1_no_kg.append(val_f1_no_kg_val)
            
            # После валидации, добавить:
            train_losses_kg.append(np.mean(epoch_train_losses_kg))
            val_losses_kg.append(np.mean(epoch_val_losses_kg))
            train_accs_kg.append(100 * train_correct_kg / train_total_kg)
            val_accs_kg.append(val_acc_kg)

            train_losses_no_kg.append(np.mean(epoch_train_losses_no_kg))
            val_losses_no_kg.append(np.mean(epoch_val_losses_no_kg))
            train_accs_no_kg.append(100 * train_correct_no_kg / train_total_no_kg)
            val_accs_no_kg.append(val_acc_no_kg)

            print(f"Валидационная точность (KG): {val_acc_kg:.2f}%")
            print(f"Валидационные метрики (KG): P={val_prec_kg:.3f}, R={val_rec_kg:.3f}, F1={val_f1_kg_val:.3f}")
            print(f"Валидационная точность (без KG): {val_acc_no_kg:.2f}%")
            print(f"Валидационные метрики (без KG): P={val_prec_no_kg:.3f}, R={val_rec_no_kg:.3f}, F1={val_f1_no_kg_val:.3f}")
            
            if val_acc_kg > best_val_acc_kg:
                best_val_acc_kg = val_acc_kg
                torch.save(model_with_kg.state_dict(), 'best_contradiction_model_kg_11.pth')
            
            if val_acc_no_kg > best_val_acc_no_kg:
                best_val_acc_no_kg = val_acc_no_kg
                torch.save(model_without_kg.state_dict(), 'best_contradiction_model_no_kg_11.pth')
        
        print(f"✅ Обучение завершено.")
        print(f"Лучшая валидационная точность (KG): {best_val_acc_kg:.2f}%")
        print(f"Лучшая валидационная точность (без KG): {best_val_acc_no_kg:.2f}%")
        # Построение детальных графиков обучения
        try:
            plot_training_curves(train_losses_kg, val_losses_kg, train_accs_kg, val_accs_kg,
                                train_losses_no_kg, val_losses_no_kg, train_accs_no_kg, val_accs_no_kg,
                                train_precision_kg, val_precision_kg, train_recall_kg, val_recall_kg, train_f1_kg, val_f1_kg,
                                train_precision_no_kg, val_precision_no_kg, train_recall_no_kg, val_recall_no_kg, train_f1_no_kg, val_f1_no_kg,
                                'charts/training_curves_detailed.png')
    
            # Сводный график финальных метрик
            final_metrics_kg = [val_accs_kg[-1]/100, val_precision_kg[-1], val_recall_kg[-1], val_f1_kg[-1]]
            final_metrics_no_kg = [val_accs_no_kg[-1]/100, val_precision_no_kg[-1], val_recall_no_kg[-1], val_f1_no_kg[-1]]
            
            plot_metrics_summary(final_metrics_kg, final_metrics_no_kg, 'charts/metrics_summary.png')
    
        except Exception as e:
            print(f"⚠️ Ошибка создания графиков: {e}")
        return model_with_kg, model_without_kg
    
    def evaluate_model(self, model_with_kg, model_without_kg, val_loader):
        """Оценка обеих моделей"""
        print("Этап 6: Финальная оценка моделей")
        
        model_with_kg.eval()
        model_without_kg.eval()
        all_predictions_kg = []
        all_predictions_no_kg = []
        all_labels = []
        
        with torch.no_grad():
            eval_pbar = tqdm(val_loader, desc="Оценка моделей")
            for batch in eval_pbar:
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    kg_embeddings = batch['kg_embedding'].to(self.device)
                    
                    outputs_kg = model_with_kg(input_ids, attention_mask, kg_embeddings)
                    _, predicted_kg = torch.max(outputs_kg.data, 1)
                    all_predictions_kg.extend(predicted_kg.cpu().numpy())
                    
                    outputs_no_kg = model_without_kg(input_ids, attention_mask)
                    _, predicted_no_kg = torch.max(outputs_no_kg.data, 1)
                    all_predictions_no_kg.extend(predicted_no_kg.cpu().numpy())
                    
                    all_labels.extend(labels.cpu().numpy())
                    
                    eval_pbar.set_postfix({
                        'Обработано': f'{len(all_predictions_kg)}'
                    })
                    
                except Exception as e:
                    print(f"⚠️ Ошибка при оценке батча: {e}")
                    continue
        
        if len(all_predictions_kg) == 0 or len(all_labels) == 0:
            print("Не удалось получить предсказания для оценки")
            return 0, 0, "Нет данных для оценки"
        
        accuracy_kg = accuracy_score(all_labels, all_predictions_kg)
        report_kg = classification_report(
            all_labels,
            all_predictions_kg,
            target_names=['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO'],
            zero_division=0
        )
        
        accuracy_no_kg = accuracy_score(all_labels, all_predictions_no_kg)
        report_no_kg = classification_report(
            all_labels,
            all_predictions_no_kg,
            target_names=['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO'],
            zero_division=0
        )
        
        print(f"\n📊 Результаты оценки:")
        print(f"\nМодель с KG:")
        print(f"Точность: {accuracy_kg:.4f}")
        print(f"Количество предсказаний: {len(all_predictions_kg)}")
        print(f"Детальный отчет:\n{report_kg}")
        
        print(f"\nМодель без KG:")
        print(f"Точность: {accuracy_no_kg:.4f}")
        print(f"Количество предсказаний: {len(all_predictions_no_kg)}")
        print(f"Детальный отчет:\n{report_no_kg}")
        
        print(f"\n📈 Сравнение моделей:")
        print(f"Разница в точности (KG - без KG): {(accuracy_kg - accuracy_no_kg):.4f}")
        
        # В конце метода evaluate_model, после всех print'ов добавить:

        # Построение матриц ошибок
        class_names = ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']
        plot_confusion_matrices(all_labels, all_predictions_kg, all_predictions_no_kg,
                            class_names, 'charts/confusion_matrices_comparison.png')

        return accuracy_kg, accuracy_no_kg, f"KG:\n{report_kg}\nNo KG:\n{report_no_kg}"

def check_requirements():
    """Проверка и установка необходимых зависимостей"""
    print("Проверка зависимостей...")
    
    required_packages = {
        'torch': 'torch',
        'transformers': 'transformers',
        'spacy': 'spacy',
        'sklearn': 'scikit-learn',
        'networkx': 'networkx'
    }
    
    missing_packages = []
    
    for package, install_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(install_name)
    
    if missing_packages:
        print(f"Отсутствуют пакеты: {missing_packages}")
        print(f"pip install {' '.join(missing_packages)}")
        try:
            import spacy
            try:
                spacy.load('en_core_web_sm')
            except OSError:
                print("Загружаем spacy модель...")
                os.system("python -m spacy download en_core_web_sm")
        except ImportError:
            pass
        return False
    
    print("✅ Все зависимости установлены")
    return True

def test_fever_data_structure(data_path, num_samples=5):
    """Тестирование структуры данных FEVER"""
    print(f"Анализ структуры данных в {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"Файл {data_path} не найден!")
        return False
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                
                try:
                    item = json.loads(line.strip())
                    print(f"\nЗапись {i}:")
                    print(f"  Ключи: {list(item.keys())}")
                    print(f"  ID: {item.get('id', 'Нет')}")
                    print(f"  Label: {item.get('label', 'Нет')}")
                    print(f"  Claim: {item.get('claim', 'Нет')[:100]}...")
                    
                    if 'evidence' in item:
                        evidence = item['evidence']
                        print(f"  Evidence тип: {type(evidence)}")
                        print(f"  Evidence длина: {len(evidence) if isinstance(evidence, (list, str)) else 'N/A'}")
                        
                        if isinstance(evidence, list) and evidence:
                            print(f"  Evidence[0] тип: {type(evidence[0])}")
                            if isinstance(evidence[0], list) and evidence[0]:
                                print(f"  Evidence[0] длина: {len(evidence[0])}")
                                print(f"  Evidence[0] содержимое: {evidence[0]}")
                    
                    print("-" * 50)
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️ Ошибка JSON в строке {i}: {e}")
                    
        print("Анализ структуры завершен")
        return True
        
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        return False

def main():
    """Главная функция"""
    print("Система обнаружения противоречий с использованием графов знаний")
    print("=" * 70)
    
    if not check_requirements():
        print("Пожалуйста, установите недостающие зависимости")
        return
    
    data_path = "fever_data/fever_train.jsonl"
    
    if not os.path.exists(data_path):
        print(f"Файл {data_path} не найден!")
        print("Пожалуйста, убедитесь, что датасет FEVER находится в указанной папке.")
        
        possible_paths = [
            "fever_train.jsonl",
            "data/fever_train.jsonl",
            "../fever_data/fever_train.jsonl"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                print(f"✅ Найден файл: {data_path}")
                break
        else:
            return
    
    print("\n" + "="*50)
    if not test_fever_data_structure(data_path):
        return
    
    print("\n" + "="*50)
    pipeline = ContradictionDetectionPipeline(data_path)
    
    try:
        print("Запуск пайплайна с ограниченным datasets для тестирования...")
        model_with_kg, model_without_kg = pipeline.run_full_pipeline(max_samples=15000)
        
        if model_with_kg is not None and model_without_kg is not None:
            print("\nПайплайн успешно завершен!")
            print("Модели сохранены и готовы к использованию.")
            
            pipeline_info = {
                'num_entities': len(pipeline.entity_extractor.entity_to_id),
                'num_relations': len(pipeline.entity_extractor.relation_to_id),
                'graph_edges': pipeline.entity_extractor.knowledge_graph.number_of_edges(),
                'device_used': str(pipeline.device)
            }
            
            with open('charts/pipeline_info.json', 'w') as f:
                json.dump(pipeline_info, f, indent=2)
            
            print(f"\n📋 Информация о пайплайне:")
            print(f"  Сущностей в KG: {pipeline_info['num_entities']}")
            print(f"  Отношений в KG: {pipeline_info['num_relations']}")
            print(f"  Рёбер в графе: {pipeline_info['graph_edges']}")
            print(f"  Устройство: {pipeline_info['device_used']}")
        else:
            print("Пайплайн завершился с ошибками")
        
    except KeyboardInterrupt:
        print("\n⚠️ Выполнение прервано пользователем")
    except Exception as e:
        print(f"Ошибка при выполнении пайплайна: {str(e)}")
        import traceback
        print("\nДетальная информация об ошибке:")
        traceback.print_exc()
        
        print("\nДиагностика:")
        print(f"GPU доступен: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU устройство: {torch.cuda.get_device_name()}")
            print(f"GPU память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def run_quick_test():
    """Быстрый тест основных компонентов"""
    print("Запуск быстрого теста компонентов...")
    
    test_data = [
        {
            'id': 1,
            'claim': 'The capital of France is Paris.',
            'evidence': 'Paris is the capital and most populous city of France.',
            'label': 0,
            'label_name': 'SUPPORTS'
        },
        {
            'id': 2,
            'claim': 'The Earth is flat.',
            'evidence': 'The Earth is a sphere.',
            'label': 1,
            'label_name': 'REFUTES'
        }
    ]
    
    try:
        extractor = EntityExtractor()
        entity_pairs = extractor.extract_entities_and_relations(test_data)
        print(f"Тест извлечения сущностей прошел успешно")
        print(f"Найдено сущностей: {len(extractor.entity_to_id)}")
    except Exception as e:
        print(f"Тест извлечения сущностей неуспешен: {e}")
    
    try:
        model = ContradictionClassifier('bert-base-uncased')
        print(f"Тест создания модели прошел успешно")
    except Exception as e:
        print(f"Тест создания модели неуспешен: {e}")
    
    print("Быстрый тест завершен")

if __name__ == "__main__":
    main()
