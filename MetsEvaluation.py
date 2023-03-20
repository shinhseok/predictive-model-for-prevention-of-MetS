import pandas as pd
import numpy as np
import scipy.stats

from sklearn.metrics import confusion_matrix, auc, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve, CalibrationDisplay, CalibratedClassifierCV

from matplotlib import pyplot as plt

def get_feature_importance(feature_importance, col_names, top_n, is_abs=False) :
    if is_abs :
        feature_importance = abs(feature_importance)
    idx = feature_importance != 0
    col_name = col_names[idx]
    importance = feature_importance[idx]
    result = pd.DataFrame({'feature' : col_name, 'importance' : importance}).sort_values(by='importance', ascending=False)[:top_n]
    return result

def get_metric(prob, label, threshold):
    prob = prob[:,1]
    prd = np.where(prob>=threshold, 1, 0)
    
    tn, fp, fn, tp = confusion_matrix(label, prd, labels=[0,1]).ravel()
    
    accuracy = (tn+tp)/(tn+fp+fn+tp)
    sensitivity = tp/(fn+tp)
    specificity = tn/(fp+tn)
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)
    f1 = 2*(sensitivity*ppv)/(sensitivity+ppv)
    balaced_accuracy = 0.5*(sensitivity+specificity)
    fpr, tpr, thresholds = roc_curve(label, prob, pos_label=1)
    auc_roc = auc(fpr,tpr)
    
    res = {'acc' : accuracy,
           'bac' : balaced_accuracy,
           'recall': sensitivity,
           'ppv':ppv,
           'npv':npv,
           'sepecificity':specificity,
           'f1':f1,
           'auc':auc_roc}
    
    return res

def get_calib_prob(prob, label, beta):    
    return beta*prob/(beta*prob-prob+1)


def get_z_test(y_test, y_hat):
    a = np.sum((y_test-y_hat)*(1-2*y_hat))
    b = np.sum((1-2*y_hat)**2*y_hat*(1-y_hat))**0.5
    return a/b


def get_calib_metric(prob, label, beta, tau):
    
    prob = beta*prob/(beta*prob-prob+1)
    threshold = tau
    
    #z-static
    a = np.sum((label-prob)*(1-2*prob))
    b = np.sum((1-2*prob)**2*prob*(1-prob))**0.5
    z_static = a/b
    
    prd = np.where(prob>=threshold, 1, 0)
    
    tn, fp, fn, tp = confusion_matrix(label, prd, labels=[0,1]).ravel()
    
    accuracy = (tn+tp)/(tn+fp+fn+tp)
    sensitivity = tp/(fn+tp)
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)
    specificity = tn/(fp+tn)
    f1 = 2*(sensitivity*ppv)/(sensitivity+ppv)
    balaced_accuracy = 0.5*(sensitivity+specificity)
    fpr, tpr, thresholds = roc_curve(label, prob, pos_label=1)
    
    print('==== Calibration ====')
    print('Accuracy :', accuracy)
    print('Balanced Accuracy :', balaced_accuracy)
    print('Sensitivity : ', sensitivity)
    print('PPV : ', ppv)
    print('NPV : ', npv)
    print('Specificity : ', specificity)
    print('F1 score : ', f1)
    print('AUC : ', auc(fpr,tpr))
    
    return prob, z_static


def calibrated_plot(model_name, model, X_train, y_train, X_valid, y_valid, X_test, y_test, beta, tau, bins=10, is_tabnet=False):

    # calibrated
    calibrator_sigmoid = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    calibrator_isotonic = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    
    if is_tabnet:
        y_prob = model.predict_proba(X_test.values[:])[:,1]
        calibrator_sigmoid.fit(X_valid.values[:], y_valid.values[:])
        calibrator_isotonic.fit(X_valid.values[:], y_valid.values[:])
        y_sigmoid = calibrator_sigmoid.predict_proba(X_test.values[:])[:,1]
        y_isotonic = calibrator_isotonic.predict_proba(X_test.values[:])[:,1]
        
    else :
        y_prob = model.predict_proba(X_test)[:,1]
        calibrator_sigmoid.fit(X_valid, y_valid)
        calibrator_isotonic.fit(X_valid, y_valid)
        y_sigmoid = calibrator_sigmoid.predict_proba(X_test)[:,1]
        y_isotonic = calibrator_isotonic.predict_proba(X_test)[:,1]
    
    y_undersample = get_calib_metric(y_prob, y_test, beta, tau)[0]
    
    brier_score_origin = brier_score_loss(y_test, y_prob)
    brier_score_sigmoid = brier_score_loss(y_test, y_sigmoid)
    brier_score_isotonic = brier_score_loss(y_test, y_isotonic)
    brier_score_undersample = brier_score_loss(y_test, y_undersample)
    
    origin_true, prob_pred_uncalibrated = calibration_curve(y_test, y_prob, n_bins=bins)
    sigmoid_true_calibrated, y_sigmoid_calibrated = calibration_curve(y_test, y_sigmoid, n_bins=bins)
    isotonic_true_calibrated, y_isotonic_calibrated = calibration_curve(y_test, y_isotonic, n_bins=bins)
    undersample_true_calibrated, y_undersample_calibrated = calibration_curve(y_test, y_undersample, n_bins=bins)
    
    #z-static
    z_origin = get_z_test(y_test, y_prob)
    z_sigmoid = get_z_test(y_test, y_sigmoid)
    z_isotonic = get_z_test(y_test, y_isotonic)
    z_undersample = get_z_test(y_test, y_undersample)
    
    p_origin = scipy.stats.norm.sf(abs(z_origin))*2
    p_sigmoid = scipy.stats.norm.sf(abs(z_sigmoid))*2
    p_isotonic = scipy.stats.norm.sf(abs(z_isotonic))*2
    p_undersample = scipy.stats.norm.sf(abs(z_undersample))*2
    
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    # plot model reliabilities
    plt.title(model_name) #'Calibration Plot '
    plt.plot(prob_pred_uncalibrated, origin_true, color = 'c', alpha=0.5, marker='o', label='Original')
    plt.plot(y_sigmoid_calibrated, sigmoid_true_calibrated, color = 'm', alpha=0.4, marker='s', label='sigmoid')
    plt.plot(y_isotonic_calibrated, isotonic_true_calibrated, color = 'b', alpha=0.4, marker='x', label='Isotonic')
    plt.plot(y_undersample_calibrated, undersample_true_calibrated, color = 'g', alpha=0.4, marker='^', label='Undersample')
    plt.xlabel('Mean predicted value')
    plt.ylabel('Fraction of positives')
    plt.legend()
    plt.savefig('./fig/'+model_name+'_calibration.png')
    plt.show()
    
    plt.title('Probability Histogram: '+model_name)
    plt.hist(y_prob, bins=10, color='c', alpha=0.2, label='Original')
    plt.hist(y_sigmoid, bins=10, color='m', alpha=0.5, label='Sigmoid', histtype='step')
    plt.hist(y_isotonic, bins=10, color='b', alpha=0.5, label='Isotonic', histtype='step')
    plt.hist(y_undersample, bins=10, color='g', alpha=0.5, label='Undersample', histtype='step')
    plt.xlabel('Predictied Probabilty')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('./fig/'+model_name+'_histogram.png')
    plt.show()
    
    brier_score = {'original': brier_score_origin,
                   'sigmoid':brier_score_sigmoid,
                   'isotonic':brier_score_isotonic,
                   'underample':brier_score_undersample }
    
    z_static = {'original': z_origin,
                'sigmoid':z_sigmoid,
                'isotonic':z_isotonic,
                'underample':z_undersample }
    
    p_value = {'original': p_origin,
                'sigmoid':p_sigmoid,
                'isotonic':p_isotonic,
                'underample':p_undersample }
    
    return brier_score, z_static, p_value