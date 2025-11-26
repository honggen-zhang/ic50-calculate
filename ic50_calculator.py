import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class IC50Calculator:
    """
    Enhanced IC50 calculation tool with support for multiple models and statistical analysis.
    """
    
    def __init__(self, model='4PL'):
        """
        Initialize IC50 calculator.
        
        Parameters:
        -----------
        model : str, optional
            Model type: '3PL', '4PL', or '5PL' (default: '4PL')
        """
        self.model = model.upper()
        self.params = None
        self.covariance = None
        self.r_squared = None
        self.ic50 = None
        self.ic50_ci = None
        self.outliers = None
        self.weights = None
        self.response_mean = None  # Mean of multiple measurements
        self.response_std = None   # Std of multiple measurements
        self.has_multiple_measurements = False  # Flag for multiple measurements
        
    def four_pl(self, x, bottom, top, logIC50, hill_slope):
        """4-Parameter Logistic model"""
        return bottom + (top - bottom) / (1 + 10**((logIC50 - np.log10(x)) * hill_slope))
    
    def three_pl(self, x, bottom, top, logIC50):
        """3-Parameter Logistic model (fixed hill slope = -1)"""
        return bottom + (top - bottom) / (1 + 10**((logIC50 - np.log10(x)) * (-1)))
    
    def five_pl(self, x, bottom, top, logIC50, hill_slope, asymmetry):
        """5-Parameter Logistic model"""
        return bottom + (top - bottom) / (1 + (10**((logIC50 - np.log10(x)) * hill_slope))**asymmetry)
    
    def _process_multiple_measurements(self, responses):
        # Check if responses is a list/array of arrays
        if isinstance(responses, (list, tuple)) and len(responses) > 0:
            # Check if first element is array-like
            if isinstance(responses[0], (list, tuple, np.ndarray)):
                # Multiple measurements provided
                responses_array = np.array(responses)
                if responses_array.ndim == 2:
                    # Calculate mean and std across measurements
                    response_mean = np.mean(responses_array, axis=0)
                    response_std = np.std(responses_array, axis=0, ddof=1)  # Sample std
                    return response_mean, response_std, True
        
        # Single measurement or already processed
        responses_array = np.array(responses)
        return responses_array, None, False
    
    def calculate_r_squared(self, y_observed, y_predicted):
        """Calculate R-squared value"""
        ss_res = np.sum((y_observed - y_predicted) ** 2)
        ss_tot = np.sum((y_observed - np.mean(y_observed)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def calculate_ic50_ci(self, confidence=0.95):
        """Calculate confidence interval for IC50"""
        if self.covariance is None:
            return None
        
        alpha = 1 - confidence
        df = len(self.params) - 1  # degrees of freedom
        logIC50_idx = 2
        
        logIC50_se = np.sqrt(self.covariance[logIC50_idx, logIC50_idx])
        t_value = stats.t.ppf(1 - alpha/2, df)
        
        logIC50_lower = self.params[logIC50_idx] - t_value * logIC50_se
        logIC50_upper = self.params[logIC50_idx] + t_value * logIC50_se
        
        ic50_lower = 10**logIC50_lower
        ic50_upper = 10**logIC50_upper
        
        return (ic50_lower, ic50_upper)
    
    def detect_outliers(self, concentrations, responses, method='cooks', threshold=None):
        conc = np.array(concentrations)
        resp = np.array(responses)
        
        # First, do a preliminary fit to get residuals
        try:
            # Quick fit without bounds for outlier detection
            if self.model == '3PL':
                func = self.three_pl
                p0 = [0, 100, np.log10(np.median(conc[conc > 0]))]
            elif self.model == '4PL':
                func = self.four_pl
                p0 = [0, 100, np.log10(np.median(conc[conc > 0])), -1]
            else:  # 5PL
                func = self.five_pl
                p0 = [0, 100, np.log10(np.median(conc[conc > 0])), -1, 1]
            
            # Filter zeros
            mask = conc != 0
            conc_fit = conc[mask]
            resp_fit = resp[mask]
            
            if len(conc_fit) < 3:
                return np.zeros(len(conc), dtype=bool)
            
            # Initial fit
            try:
                params_init, _ = curve_fit(func, conc_fit, resp_fit, p0=p0, maxfev=5000)
                y_pred = func(conc_fit, *params_init)
                residuals = resp_fit - y_pred
            except:
                # If fit fails, use simple methods
                residuals = resp_fit - np.mean(resp_fit)
        except:
            residuals = resp_fit - np.mean(resp_fit) if len(resp_fit) > 0 else np.array([])
        
        outlier_mask = np.zeros(len(conc), dtype=bool)
        outlier_mask_fit = np.zeros(len(conc_fit), dtype=bool)
        
        if method == 'cooks':
            # Cook's distance
            if len(residuals) > 0 and len(conc_fit) > 3:
                mse = np.mean(residuals**2)
                if mse > 0:
                    h = 1.0 / len(conc_fit)  # Simplified leverage
                    cooks_d = (residuals**2) / (mse * (1 - h)**2)
                    threshold_cooks = threshold if threshold else 4.0 / len(conc_fit)
                    outlier_mask_fit = cooks_d > threshold_cooks
        
        elif method == 'iqr':
            # Interquartile Range method
            if len(residuals) > 0:
                Q1 = np.percentile(residuals, 25)
                Q3 = np.percentile(residuals, 75)
                IQR = Q3 - Q1
                threshold_iqr = threshold if threshold else 1.5
                lower_bound = Q1 - threshold_iqr * IQR
                upper_bound = Q3 + threshold_iqr * IQR
                outlier_mask_fit = (residuals < lower_bound) | (residuals > upper_bound)
        
        elif method == 'zscore':
            # Z-score method
            if len(residuals) > 0:
                z_scores = np.abs(stats.zscore(residuals))
                threshold_z = threshold if threshold else 3.0
                outlier_mask_fit = z_scores > threshold_z
        
        elif method == 'residual':
            # Residual-based method
            if len(residuals) > 0:
                std_residuals = np.std(residuals)
                threshold_res = threshold if threshold else 2.5
                outlier_mask_fit = np.abs(residuals) > threshold_res * std_residuals
        
        # Map back to original array
        outlier_mask[mask] = outlier_mask_fit
        
        return outlier_mask
    
    def robust_fit(self, concentrations, responses, detect_outliers=False, outlier_method='cooks', 
            remove_outliers=False, exclude_zero=True, bounds=None,
                   robust_method='iterative', max_iter=10, tolerance=1e-3):

        conc = np.array(concentrations)
        
        # Process multiple measurements if provided
        resp, resp_std, has_multiple = self._process_multiple_measurements(responses)
        self.has_multiple_measurements = has_multiple
        self.response_mean = resp
        self.response_std = resp_std
        
        # Validate input
        if len(conc) != len(resp):
            raise ValueError("Concentrations and responses must have the same length")
        if len(conc) < 3:
            raise ValueError("Need at least 3 data points for fitting")
        
        # Detect outliers if requested
        if detect_outliers or remove_outliers:
            self.outliers = self.detect_outliers(conc, resp, method=outlier_method, threshold=5)
            if remove_outliers:
                if len(conc) < 3:
                    raise ValueError("Too many outliers removed. Need at least 3 data points for fitting.")
        
        # Filter out zero concentrations if requested
        if exclude_zero:
            mask = conc != 0
            conc_fit = conc[mask]
            resp_fit = resp[mask]
        else:
            mask = np.ones(len(conc), dtype=bool)
            conc_fit = conc
            resp_fit = resp
        
        if len(conc_fit) < 3:
            raise ValueError("Need at least 3 non-zero concentrations for fitting")
        
        # Select model function
        if self.model == '3PL':
            func = self.three_pl
            initial_guess = [0, 100, np.log10(np.median(conc_fit[conc_fit > 0]))]
            if bounds is None:
                bounds = ([-np.inf, 0, -10], [np.inf, 200, 10])
        elif self.model == '4PL':
            func = self.four_pl
            initial_guess = [0, 100, np.log10(np.median(conc_fit[conc_fit > 0])), -1]
            if bounds is None:
                bounds = ([-np.inf, 0, -10, -5], [np.inf, 200, 10, 0])
        elif self.model == '5PL':
            func = self.five_pl
            initial_guess = [0, 100, np.log10(np.median(conc_fit[conc_fit > 0])), -1, 1]
            if bounds is None:
                bounds = ([-np.inf, 0, -10, -5, 0.1], [np.inf, 200, 10, 0, 10])
        else:
            raise ValueError(f"Unknown model: {self.model}. Choose '3PL', '4PL', or '5PL'")
        
        if robust_method == 'iterative':
            # Iterative reweighting (IRLS - Iteratively Reweighted Least Squares)
            weights = np.ones(len(conc_fit))
            params_old = None
            
            for iteration in range(max_iter):
                try:
                    # Fit with current weights
                    self.params, self.covariance = curve_fit(
                        func, conc_fit, resp_fit,
                        p0=initial_guess,
                        bounds=bounds,
                        maxfev=10000,
                        method='trf',
                        sigma=1.0/np.sqrt(weights + 1e-10)  # Inverse of weights as sigma
                    )
                    
                    # Calculate residuals
                    y_pred = func(conc_fit, *self.params)
                    residuals = resp_fit - y_pred
                    
                    # Calculate robust weights using Huber's method
                    mad = np.median(np.abs(residuals - np.median(residuals)))
                    if mad < 1e-10:
                        mad = np.std(residuals)
                    if mad < 1e-10:
                        break
                    
                    # Huber's weight function
                    c = 1.345 * mad  # Tuning constant
                    standardized_residuals = np.abs(residuals) / mad
                    weights = np.where(standardized_residuals <= c, 
                                       np.ones_like(standardized_residuals),
                                       c / standardized_residuals)
                    
                    # Check convergence
                    if params_old is not None:
                        param_change = np.max(np.abs(self.params - params_old) / (np.abs(params_old) + 1e-10))
                        if param_change < tolerance:
                            break
                    
                    params_old = self.params.copy()
                    initial_guess = self.params  # Use current params as next guess
                    
                except Exception as e:
                    if iteration == 0:
                        raise RuntimeError(f"Robust fitting failed: {str(e)}")
                    break
            
            self.weights = np.zeros(len(conc))
            self.weights[mask] = weights
            
        elif robust_method == 'huber':
            # Huber loss function (using scipy.optimize.minimize)
            def loss_function(params):
                y_pred = func(conc_fit, *params)
                residuals = resp_fit - y_pred
                # Huber loss: quadratic for small errors, linear for large errors
                delta = 1.35  # Tuning parameter
                loss = np.sum(np.where(np.abs(residuals) <= delta,
                                     0.5 * residuals**2,
                                     delta * (np.abs(residuals) - 0.5 * delta)))
                return loss
            
            try:
                result = minimize(loss_function, initial_guess, method='L-BFGS-B', bounds=bounds)
                self.params = result.x
                # Approximate covariance (may not be accurate for robust methods)
                y_pred = func(conc_fit, *self.params)
                residuals = resp_fit - y_pred
                sse = np.sum(residuals**2)
                self.covariance = np.eye(len(self.params)) * sse / (len(conc_fit) - len(self.params))
            except Exception as e:
                raise RuntimeError(f"Huber loss fitting failed: {str(e)}")

        logIC50 = self.params[2]
        
        self.ic50 = 10**logIC50
        
        # Calculate R-squared
        y_pred = func(conc_fit, *self.params)
        self.r_squared = self.calculate_r_squared(resp_fit, y_pred)
        
        # Calculate confidence interval
        try:
            self.ic50_ci = self.calculate_ic50_ci()
        except:
            self.ic50_ci = None
        
        # Return results
        results = {
            'IC50': self.ic50,
            'IC50_CI': self.ic50_ci,
            'R_squared': self.r_squared,
            'parameters': self.params,
            'model': self.model,
            'robust_method': robust_method
        }
        
        # Add multiple measurement statistics if available
        if self.has_multiple_measurements:
            results['has_multiple_measurements'] = True
            if self.response_std is not None:
                results['response_mean'] = self.response_mean
                results['response_std'] = self.response_std
        
        if self.model == '4PL':
            results['bottom'] = self.params[0]
            results['top'] = self.params[1]
            results['hill_slope'] = self.params[3]
        elif self.model == '3PL':
            results['bottom'] = self.params[0]
            results['top'] = self.params[1]
        elif self.model == '5PL':
            results['bottom'] = self.params[0]
            results['top'] = self.params[1]
            results['hill_slope'] = self.params[3]
            results['asymmetry'] = self.params[4]
        
        return results


