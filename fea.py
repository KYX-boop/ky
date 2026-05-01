#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, confusion_matrix
import joblib
import os
import pickle
import json


class OptimizedFeatureFusion:
    def __init__(self, dl_features_path, radiomics_path, radiomics_delta_path, clinical_path):
        self.paths = {
            'dl': dl_features_path,
            'radiomics': radiomics_path,
            'radiomics_delta': radiomics_delta_path,
            'clinical': clinical_path
        }
        self.clinical_cols = ['T_stage', 'HER2_status', 'NAC_classification', 
                             'ER_status', 'PR_status', 'Ki_67']
        self.data = self._load_all_data()
        self.merged_data = None
        self.train_data = None
        self.test_sets = None

    def _load_all_data(self):
        data_dict = {}
        
        print("Loading DL features...")
        try:
            if self.paths['dl'].endswith('.pkl'):
                with open(self.paths['dl'], 'rb') as f:
                    dl_data = pickle.load(f)
                dl_df = pd.DataFrame(dl_data['features'])
                dl_df.columns = [f"feature_{i}" for i in range(dl_data['features'].shape[1])]
                dl_df['patient_ID'] = dl_data['patient_ids']
                dl_df['hospital'] = dl_data['hospitals']
                dl_df['label'] = dl_data['labels']
            else:
                dl_df = pd.read_csv(self.paths['dl'], encoding='utf-8')
                if 'patient_id' in dl_df.columns:
                    dl_df = dl_df.rename(columns={'patient_id': 'patient_ID'})
                if 'bpCR' in dl_df.columns and 'label' not in dl_df.columns:
                    dl_df = dl_df.rename(columns={'bpCR': 'label'})
            
            dl_df['patient_ID'] = dl_df['patient_ID'].astype(str).str.strip()
            data_dict['dl'] = dl_df
            
            feature_cols = [col for col in dl_df.columns if col.startswith('feature_')]
            print(f"DL features: {len(feature_cols)} dims, {len(dl_df)} samples")
            
        except Exception as e:
            print(f"Error loading DL features: {e}")
            raise
        
        for feat_type, path in [('radiomics', self.paths['radiomics']), 
                               ('radiomics_delta', self.paths['radiomics_delta']),
                               ('clinical', self.paths['clinical'])]:
            try:
                if feat_type == 'radiomics_delta' and not os.path.exists(path):
                    continue
                    
                df = pd.read_csv(path, encoding='gbk')
                df['patient_ID'] = df['patient_ID'].astype(str).str.strip()
                data_dict[feat_type] = df
                print(f"{feat_type}: {df.shape}")
                
            except Exception as e:
                print(f"Warning: {feat_type} load failed: {e}")
                if feat_type == 'clinical':
                    clinical_df = data_dict['dl'][['patient_ID', 'hospital', 'label']].copy()
                    clinical_df = clinical_df.rename(columns={'label': 'bpCR'})
                    for col in self.clinical_cols:
                        clinical_df[col] = 0
                    data_dict['clinical'] = clinical_df
                    print(f"Clinical data constructed: {clinical_df.shape}")
                else:
                    data_dict[feat_type] = pd.DataFrame(columns=['patient_ID'])
        
        return data_dict

    def merge_all_features(self):
        print("\nMerging features...")
        
        merged_df = self.data['clinical'][['patient_ID', 'hospital', 'bpCR'] + 
                                        [col for col in self.clinical_cols if col in self.data['clinical'].columns]].copy()
        
        print(f"Base data: {merged_df.shape}")
        
        for feat_type, feat_name in [('dl', 'DL'), ('radiomics', 'Radiomics'), ('radiomics_delta', 'Delta')]:
            if feat_type in self.data and len(self.data[feat_type]) > 0:
                feat_df = self.data[feat_type]
                
                if feat_type == 'dl':
                    cols = ['patient_ID'] + [c for c in feat_df.columns if c.startswith('feature_')]
                else:
                    cols = [c for c in feat_df.columns if c not in ['hospital', 'bpCR', 'label']]
                
                feat_subset = feat_df[cols]
                before_merge = len(merged_df)
                merged_df = pd.merge(merged_df, feat_subset, on='patient_ID', how='inner')
                after_merge = len(merged_df)
                
                print(f"{feat_name}: +{len(cols)-1} features, {before_merge}→{after_merge} samples")
        
        print(f"\nMerged: {merged_df.shape[0]} samples, {merged_df.shape[1]-3} features")
        
        label_dist = merged_df['bpCR'].value_counts()
        print(f"Label distribution: {label_dist.to_dict()}")
        
        self.merged_data = merged_df
        return merged_df

    def split_by_hospital(self, train_hospitals=[3, 5], test_hospitals=[1, 2, 4]):
        if self.merged_data is None:
            self.merge_all_features()
        
        train_df = self.merged_data[self.merged_data['hospital'].isin(train_hospitals)]
        X_train = train_df.drop(columns=['patient_ID', 'hospital', 'bpCR'])
        y_train = train_df['bpCR'].astype(int)
        
        test_sets = {}
        for hospital in test_hospitals:
            test_df = self.merged_data[self.merged_data['hospital'] == hospital]
            if len(test_df) > 0:
                X_test = test_df.drop(columns=['patient_ID', 'hospital', 'bpCR'])
                y_test = test_df['bpCR'].astype(int)
                test_sets[f'test_hospital_{hospital}'] = (X_test, y_test)
        
        print(f"\nData split:")
        print(f"Train: {len(X_train)} samples, labels: {y_train.value_counts().to_dict()}")
        for hosp_name, (X, y) in test_sets.items():
            print(f"{hosp_name}: {len(X)} samples, labels: {y.value_counts().to_dict()}")
        
        self.train_data = (X_train, y_train)
        self.test_sets = test_sets
        return (X_train, y_train), test_sets, self.merged_data

    def smart_feature_selection(self, X_train=None, y_train=None, test_sets=None):
        if X_train is None or y_train is None or test_sets is None:
            if self.train_data is None:
                self.split_by_hospital()
            X_train, y_train = self.train_data
            test_sets = self.test_sets
        
        print(f"\nSmart feature selection...")
        print(f"Initial features: {X_train.shape[1]}")
        
        dl_cols = [col for col in X_train.columns if col.startswith('feature_')]
        radio_cols = [col for col in X_train.columns if any(col.startswith(p) for p in ['preDCE_', 'postDCE_', 'preDWI_', 'postDWI_']) and 'delta_' not in col]
        delta_cols = [col for col in X_train.columns if 'delta_' in col]
        clin_cols = [col for col in X_train.columns if col in self.clinical_cols]
        
        print(f"\nFeature groups:")
        print(f"DL: {len(dl_cols)}")
        print(f"Radiomics: {len(radio_cols)}")
        print(f"Delta: {len(delta_cols)}")
        print(f"Clinical: {len(clin_cols)}")
        
        print(f"\n1. DL feature selection (target: 15-25):")
        selected_dl = []
        
        if len(dl_cols) > 0:
            dl_scores = []
            alpha_corrected = 0.05 / len(dl_cols)
            
            for col in tqdm(dl_cols, desc="Evaluating DL"):
                try:
                    stat, pvalue = ttest_ind(
                        X_train[col][y_train==0].dropna(),
                        X_train[col][y_train==1].dropna()
                    )
                    
                    mean0 = X_train[col][y_train==0].mean()
                    mean1 = X_train[col][y_train==1].mean()
                    std_pooled = np.sqrt(((X_train[col][y_train==0].var() + X_train[col][y_train==1].var()) / 2))
                    effect_size = abs(mean1 - mean0) / std_pooled if std_pooled > 0 else 0
                    
                    dl_scores.append((col, pvalue, effect_size))
                except:
                    dl_scores.append((col, 1.0, 0.0))
            
            dl_scores.sort(key=lambda x: (x[1], -x[2]))
            
            target_dl_features = 20
            good_dl = [col for col, pval, effect in dl_scores if pval < alpha_corrected]
            
            if len(good_dl) > target_dl_features:
                selected_dl = [col for col, _, _ in dl_scores[:target_dl_features]]
            else:
                selected_dl = good_dl
            
            print(f"Retained: {len(selected_dl)} DL features (alpha={alpha_corrected:.2e})")
        
        print(f"\n2. Radiomics feature selection:")
        selected_others = []
        
        other_cols = radio_cols + delta_cols
        if len(other_cols) > 0:
            print(f"Processing {len(other_cols)} radiomics features...")
            
            other_data = X_train[other_cols].fillna(0)
            variances = other_data.var()
            
            var_threshold = variances.quantile(0.75)
            high_var_cols = variances[variances > var_threshold].index.tolist()
            print(f"Variance filtering: {len(other_cols)} → {len(high_var_cols)}")
            
            if len(high_var_cols) > 0:
                try:
                    k = min(20, len(high_var_cols))
                    selector = SelectKBest(f_classif, k=k)
                    selector.fit(X_train[high_var_cols], y_train)
                    
                    significant_others = [high_var_cols[i] for i in selector.get_support(indices=True)]
                    print(f"Statistical filtering: {len(high_var_cols)} → {len(significant_others)}")
                    
                except Exception as e:
                    print(f"Warning: Statistical filtering failed: {e}")
                    significant_others = high_var_cols[:min(15, len(high_var_cols))]
            else:
                significant_others = []
            
            if len(significant_others) > 12:
                print(f"Decorrelation...")
                
                sample_size = min(30, len(significant_others))
                sampled_features = np.random.choice(significant_others, sample_size, replace=False)
                
                try:
                    corr_matrix = X_train[sampled_features].corr().abs()
                    
                    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.85)]
                    
                    selected_others = [col for col in sampled_features if col not in to_drop][:12]
                    print(f"Decorrelation: {len(significant_others)} → {len(selected_others)}")
                    
                except Exception as e:
                    print(f"Warning: Decorrelation failed: {e}")
                    selected_others = significant_others[:12]
            else:
                selected_others = significant_others
                
            print(f"Final radiomics selection: {len(selected_others)}")
        
        print(f"\n3. Clinical features: all {len(clin_cols)}")
        
        all_selected = selected_dl + selected_others + clin_cols
        print(f"\n4. Feature merging:")
        print(f"DL: {len(selected_dl)}")
        print(f"Others: {len(selected_others)}")
        print(f"Clinical: {len(clin_cols)}")
        print(f"Total: {len(all_selected)}")
        
        if len(all_selected) > 60:
            print(f"\n5. Ensemble feature selection...")
            
            methods_votes = {}
            
            try:
                catboost = CatBoostClassifier(iterations=500, verbose=0, random_state=None)
                catboost.fit(X_train[all_selected], y_train)
                cb_importance = catboost.feature_importances_
                cb_ranking = np.argsort(cb_importance)[::-1]
                
                for i, idx in enumerate(cb_ranking):
                    col = all_selected[idx]
                    methods_votes[col] = methods_votes.get(col, 0) + (len(all_selected) - i)
            except:
                pass
            
            try:
                selector = SelectKBest(f_classif, k=min(50, len(all_selected)))
                selector.fit(X_train[all_selected], y_train)
                scores = selector.scores_
                kb_ranking = np.argsort(scores)[::-1]
                
                for i, idx in enumerate(kb_ranking):
                    col = all_selected[idx]
                    methods_votes[col] = methods_votes.get(col, 0) + (len(all_selected) - i)
            except:
                pass
            
            voted_features = sorted(methods_votes.items(), key=lambda x: x[1], reverse=True)
            
            final_features = []
            dl_count = 0
            target_dl = min(15, len(selected_dl))
            
            for col, votes in voted_features:
                if len(final_features) >= 50:
                    break
                    
                if col.startswith('feature_'):
                    if dl_count < target_dl:
                        final_features.append(col)
                        dl_count += 1
                elif col in clin_cols:
                    final_features.append(col)
                elif len(final_features) < 45:
                    final_features.append(col)
        else:
            final_features = all_selected
        
        print(f"\nFinal feature selection:")
        final_dl = [col for col in final_features if col.startswith('feature_')]
        final_others = [col for col in final_features if not col.startswith('feature_') and col not in clin_cols]
        final_clin = [col for col in final_features if col in clin_cols]
        
        print(f"DL: {len(final_dl)}")
        print(f"Others: {len(final_others)}")
        print(f"Clinical: {len(final_clin)}")
        print(f"Total: {len(final_features)}")
        
        print(f"\nTraining final model with cross-validation...")
        
        best_cv_auc = 0
        best_model = None
        best_C = None
        cv_results = []
        
        for C in [0.1, 1.0, 10.0]:
            model = LogisticRegression(
                max_iter=10000,
                penalty='l2',
                C=C,
                class_weight='balanced',
                random_state=None
            )
            
            cv_scores = cross_val_score(model, X_train[final_features], y_train, 
                                       cv=5, scoring='roc_auc')
            mean_cv_auc = cv_scores.mean()
            std_cv_auc = cv_scores.std()
            
            cv_results.append({
                'C': C,
                'mean_auc': mean_cv_auc,
                'std_auc': std_cv_auc,
                'cv_scores': cv_scores.tolist()
            })
            
            print(f"C={C}: CV AUC = {mean_cv_auc:.4f} ± {std_cv_auc:.4f}")
            
            if mean_cv_auc > best_cv_auc:
                best_cv_auc = mean_cv_auc
                best_C = C
        
        print(f"\nBest C: {best_C}, Best CV AUC: {best_cv_auc:.4f}")
        
        best_model = LogisticRegression(
            max_iter=10000,
            penalty='l2',
            C=best_C,
            class_weight='balanced',
            random_state=None
        ).fit(X_train[final_features], y_train)
        
        train_pred = best_model.predict(X_train[final_features])
        train_proba = best_model.predict_proba(X_train[final_features])[:, 1]
        
        cm = confusion_matrix(y_train, train_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = cm.ravel()[0], 0, 0, cm.ravel()[1] if len(cm.ravel()) == 2 else 0, 0
        
        train_metrics = {
            'AUC': roc_auc_score(y_train, train_proba),
            'ACC': accuracy_score(y_train, train_pred),
            'SEN': recall_score(y_train, train_pred, zero_division=0),
            'SPE': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'PPV': precision_score(y_train, train_pred, zero_division=0),
            'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0
        }
        
        test_results = {}
        for name, (X_test, y_test) in test_sets.items():
            try:
                X_test_filtered = X_test[final_features]
                y_pred = best_model.predict(X_test_filtered)
                y_proba = best_model.predict_proba(X_test_filtered)[:, 1]
                
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0
                
                test_results[name] = {
                    'AUC': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
                    'ACC': accuracy_score(y_test, y_pred),
                    'SEN': recall_score(y_test, y_pred, zero_division=0) if len(np.unique(y_test)) > 1 else 0,
                    'SPE': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'PPV': precision_score(y_test, y_pred, zero_division=0) if len(np.unique(y_test)) > 1 else 0,
                    'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0
                }
            except Exception as e:
                print(f"Warning: {name} evaluation failed: {e}")
                test_results[name] = {k: 0 for k in ['AUC', 'ACC', 'SEN', 'SPE', 'PPV', 'NPV']}
        
        feature_importance_df = pd.DataFrame({
            'feature': final_features,
            'importance': np.abs(best_model.coef_[0]),
            'type': ['DL' if col.startswith('feature_') else 'Clinical' if col in clin_cols else 'Other' for col in final_features]
        }).sort_values('importance', ascending=False)
        
        return final_features, best_model, test_results, train_metrics, feature_importance_df, cv_results


def main():
    print("Feature Fusion and Selection System")
    print("=" * 80)
    
    dl_features_path = "./extracted_features/features.csv"
    radiomics_path = "./radiomics_features.csv"
    radiomics_delta_path = "./radiomics_features_delta.csv"
    clinical_path = "./clinical_data.csv"
    
    if not os.path.exists(dl_features_path):
        print(f"ERROR: DL features not found: {dl_features_path}")
        return
    
    try:
        fusion = OptimizedFeatureFusion(
            dl_features_path, 
            radiomics_path, 
            radiomics_delta_path,
            clinical_path
        )
        
        (X_train, y_train), test_sets, merged_data = fusion.split_by_hospital()
        
        final_features, model, test_results, train_metrics, feature_importance, cv_results = fusion.smart_feature_selection()
        
        print("\n" + "=" * 80)
        print("Training Results:")
        print("=" * 80)
        
        for metric, value in train_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\n" + "=" * 80)
        print("Cross-Validation Results:")
        print("=" * 80)
        
        for cv_result in cv_results:
            print(f"\nC={cv_result['C']}: {cv_result['mean_auc']:.4f} ± {cv_result['std_auc']:.4f}")
            print(f"  5-fold scores: {[f'{s:.4f}' for s in cv_result['cv_scores']]}")
        
        print("\n" + "=" * 80)
        print("Test Results:")
        print("=" * 80)
        
        for test_name, metrics in test_results.items():
            print(f"\n{test_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        print("\n" + "=" * 80)
        print("Top 10 Important Features:")
        print("=" * 80)
        print(feature_importance.head(10).to_string())
        
        output_dir = "./results"
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(model, os.path.join(output_dir, "model.pkl"))
        feature_importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
        
        results_summary = {
            'final_features': final_features,
            'feature_count': len(final_features),
            'train_metrics': train_metrics,
            'test_results': test_results,
            'cv_results': cv_results,
            'best_C': float(model.C)
        }
        
        with open(os.path.join(output_dir, "results_summary.json"), 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()