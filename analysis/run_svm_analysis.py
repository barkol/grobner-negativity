import json, numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import os, warnings
warnings.filterwarnings('ignore')

os.chdir(r'C:/Users/karol/OneDrive/Desktop/Hidden entanglement/analysis')

with open('sim_be_classification_20260219_212426.json') as f:
    data = json.load(f)

states = data['theory']
names = [s['name'] for s in states]
labels = np.array([1 if s['is_be'] else 0 for s in states])

feature_names = ['D4', 'S2sq_S4', 'D2', 'S2', 'S4', 'G2', 'G4']
X_all = np.array([[s[fn] for fn in feature_names] for s in states])

print('Total states: {} ({} SEP + {} BE)'.format(len(states), sum(labels==0), sum(labels==1)))
print()

current_correct = sum(1 for s in states
    if (s['D4'] < -0.068 + 0.053 * s['S2sq_S4']) == s['is_be'])
print('Current 2-feature classifier (D4, S2sq_S4): {}/105'.format(current_correct))
print()

print('Misclassified by current classifier:')
for s in states:
    pred_be = s['D4'] < -0.068 + 0.053 * s['S2sq_S4']
    if pred_be != s['is_be']:
        label = 'BE' if s['is_be'] else 'SEP'
        pred = 'BE' if pred_be else 'SEP'
        print('  {:<25} true={} pred={}  D4={:.6f} S2={:.6f} G2={:.6f}'.format(s['name'], label, pred, s['D4'], s['S2'], s['G2']))
print()
print('='*70)
print('LINEAR SVM - All feature combinations')
print('='*70)
best_score = 0
best_combo = None
best_clf = None
best_scaler = None
best_combo_idx = None

for k in [2, 3, 4]:
    for combo in combinations(range(len(feature_names)), k):
        fnames = [feature_names[i] for i in combo]
        X = X_all[:, list(combo)]
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        clf = LinearSVC(C=1000, max_iter=100000, dual=True)
        clf.fit(X_sc, labels)
        score = clf.score(X_sc, labels)
        n_correct = int(score * len(labels))
        if n_correct >= 103:
            preds = clf.predict(X_sc)
            wrong = [names[i] for i in range(len(names)) if preds[i] != labels[i]]
            print('  {:<45} {}/105  wrong: {}'.format(str(fnames), n_correct, wrong))
        if n_correct > best_score:
            best_score = n_correct
            best_combo = fnames
            best_clf = clf
            best_scaler = scaler
            best_combo_idx = list(combo)

print()
print('Best linear SVM: {} -> {}/105'.format(best_combo, best_score))

print()
print('='*70)
print('RBF SVM - Best combos')
print('='*70)
for k in [2, 3, 4]:
    for combo in combinations(range(len(feature_names)), k):
        fnames = [feature_names[i] for i in combo]
        X = X_all[:, list(combo)]
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        clf = SVC(C=1000, kernel='rbf', gamma='scale')
        clf.fit(X_sc, labels)
        score = clf.score(X_sc, labels)
        n_correct = int(score * len(labels))
        if n_correct >= 104:
            preds = clf.predict(X_sc)
            wrong = [names[i] for i in range(len(names)) if preds[i] != labels[i]]
            print('  {:<45} {}/105  wrong: {}'.format(str(fnames), n_correct, wrong))

print()
print('='*70)
print('BEST LINEAR CLASSIFIER DETAILS')
print('='*70)
X_best = X_all[:, best_combo_idx]
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_best)

for C in [0.1, 1, 10, 100, 1000, 10000]:
    clf = LinearSVC(C=C, max_iter=100000, dual=True)
    clf.fit(X_sc, labels)
    n_correct = int(clf.score(X_sc, labels) * len(labels))
    preds = clf.predict(X_sc)
    wrong = [names[i] for i in range(len(names)) if preds[i] != labels[i]]
    if n_correct >= best_score - 1:
        print('  C={:<8} -> {}/105  wrong: {}'.format(C, n_correct, wrong))
        w_sc = clf.coef_[0]
        b_sc = clf.intercept_[0]
        w_orig = w_sc / scaler.scale_
        b_orig = b_sc - np.dot(w_sc * scaler.mean_, 1.0 / scaler.scale_)
        feat_str = ' + '.join('{:.6f}*{}'.format(w, fn) for w, fn in zip(w_orig, best_combo))
        print('    Decision: {} + {:.6f} < 0 => BE'.format(feat_str, b_orig))

print()
print('='*70)
print('USER-REQUESTED: D4, S2, G2 classifier')
print('='*70)
idx_d4 = feature_names.index('D4')
idx_s2 = feature_names.index('S2')
idx_g2 = feature_names.index('G2')
X_user = X_all[:, [idx_d4, idx_s2, idx_g2]]

for C in [0.1, 1, 10, 100, 1000, 10000]:
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_user)
    clf = LinearSVC(C=C, max_iter=100000, dual=True)
    clf.fit(X_sc, labels)
    n_correct = int(clf.score(X_sc, labels) * len(labels))
    preds = clf.predict(X_sc)
    wrong = [names[i] for i in range(len(names)) if preds[i] != labels[i]]
    if n_correct >= 102:
        w_sc = clf.coef_[0]
        b_sc = clf.intercept_[0]
        w_orig = w_sc / scaler.scale_
        b_orig = b_sc - np.dot(w_sc * scaler.mean_, 1.0 / scaler.scale_)
        print('  C={:<8} -> {}/105  wrong: {}'.format(C, n_correct, wrong))
        print('    w_D4={:.6f}, w_S2={:.6f}, w_G2={:.6f}, b={:.6f}'.format(w_orig[0], w_orig[1], w_orig[2], b_orig))
        print('    Decision: {:.6f}*D4 + {:.6f}*S2 + {:.6f}*G2 + {:.6f} < 0 => BE'.format(w_orig[0], w_orig[1], w_orig[2], b_orig))

print()
print('='*70)
print('D4, G2 classifier (2 features)')
print('='*70)
X_dg = X_all[:, [idx_d4, idx_g2]]

for C in [0.1, 1, 10, 100, 1000, 10000]:
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_dg)
    clf = LinearSVC(C=C, max_iter=100000, dual=True)
    clf.fit(X_sc, labels)
    n_correct = int(clf.score(X_sc, labels) * len(labels))
    preds = clf.predict(X_sc)
    wrong = [names[i] for i in range(len(names)) if preds[i] != labels[i]]
    if n_correct >= 102:
        w_sc = clf.coef_[0]
        b_sc = clf.intercept_[0]
        w_orig = w_sc / scaler.scale_
        b_orig = b_sc - np.dot(w_sc * scaler.mean_, 1.0 / scaler.scale_)
        print('  C={:<8} -> {}/105  wrong: {}'.format(C, n_correct, wrong))
        print('    Decision: {:.6f}*D4 + {:.6f}*G2 + {:.6f} < 0 => BE'.format(w_orig[0], w_orig[1], b_orig))

print()
print('='*70)
print('D4, S2sq_S4, G2 classifier')
print('='*70)
idx_s2s4 = feature_names.index('S2sq_S4')
X_dsg = X_all[:, [idx_d4, idx_s2s4, idx_g2]]

for C in [0.1, 1, 10, 100, 1000, 10000]:
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_dsg)
    clf = LinearSVC(C=C, max_iter=100000, dual=True)
    clf.fit(X_sc, labels)
    n_correct = int(clf.score(X_sc, labels) * len(labels))
    preds = clf.predict(X_sc)
    wrong = [names[i] for i in range(len(names)) if preds[i] != labels[i]]
    if n_correct >= 102:
        w_sc = clf.coef_[0]
        b_sc = clf.intercept_[0]
        w_orig = w_sc / scaler.scale_
        b_orig = b_sc - np.dot(w_sc * scaler.mean_, 1.0 / scaler.scale_)
        print('  C={:<8} -> {}/105  wrong: {}'.format(C, n_correct, wrong))
        print('    Decision: {:.6f}*D4 + {:.6f}*(S2sq_S4) + {:.6f}*G2 + {:.6f} < 0 => BE'.format(w_orig[0], w_orig[1], w_orig[2], b_orig))

print()
print('='*70)
print('chess_L feature analysis')
print('='*70)
chess_L = [s for s in states if s['name'] == 'chess_L'][0]
print('chess_L: D4={:.6f}  S2={:.6f}  G2={:.6f}  S2sq_S4={:.6f}  D2={:.6f}'.format(chess_L['D4'], chess_L['S2'], chess_L['G2'], chess_L['S2sq_S4'], chess_L['D2']))
print()

sep_states = [s for s in states if not s['is_be']]
for fn in ['D4', 'G2', 'S2', 'D2']:
    vals_sep = sorted([(s[fn], s['name']) for s in sep_states], reverse=True)[:3]
    print('  Top 3 SEP by {}: {}'.format(fn, [('{:.4f}'.format(v), n) for v, n in vals_sep]))
    print('  chess_L {} = {:.4f}'.format(fn, chess_L[fn]))
print()

print('States that are hardest to classify (margin analysis):')
for combo_name, combo_idx in [
    ('D4,S2sq_S4', [0, 1]),
    ('D4,G2', [0, 5]),
    ('D4,S2,G2', [0, 3, 5]),
    ('D4,S2sq_S4,G2', [0, 1, 5]),
    ('D4,S4,G2', [0, 4, 5]),
]:
    X = X_all[:, combo_idx]
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    clf = SVC(C=1000, kernel='linear')
    clf.fit(X_sc, labels)
    distances = clf.decision_function(X_sc)
    be_margins = [(distances[i], names[i]) for i in range(len(names)) if labels[i] == 1]
    be_margins.sort()
    print('  {}: hardest BE: {}'.format(combo_name, [('{:.4f}'.format(d), n) for d, n in be_margins[:3]]))
