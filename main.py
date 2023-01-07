import numpy as np
import pandas as pd
import scipy.stats

from matplotlib import pyplot as plt
from functions import print_table
from functions import check_distribution_hypothesis

FILE_PATH = 'data/A2.txt'


class DataStructure:
    def __init__(self, filepath: str, delimiter: str = ',', encoding: str = 'utf-8', header=None):
        self.df = pd.read_csv(filepath, sep=delimiter, encoding=encoding, header=header)
        self.n = self.df.shape[0]
        self.k = self.df.shape[1]

    def get_mean(self) -> pd.Series:
        return self.df.mean()

    def get_dispersion(self) -> pd.Series:
        return np.square(self.df.std(ddof=0))

    def get_mode(self) -> pd.DataFrame:
        return self.df.mode()

    def get_median(self) -> pd.Series:
        return self.df.median()

    def get_skewness(self) -> pd.Series:
        return self.df.skew()

    def get_kurtosis(self) -> pd.Series:
        return self.df.kurtosis()

    def get_histogram(self, bins_number: int = 25, percentage_flag: bool = True):
        weights = np.ones(self.n) / self.n if percentage_flag else None
        for col in self.df.columns:
            x = self.df[col]
            fig, ax = plt.subplots(figsize=(12, 2.8), layout='constrained')
            ax.hist(x, bins=bins_number, weights=weights)
            ax.set_xlabel('Voltage')
            ax.set_title(f"Parameter {col + 1}")
            plt.show()

    def zero_one_transformation(self, epsilon: float = 1e-8) -> pd.DataFrame:
        data = self.df.copy()
        data -= (1 + epsilon) * np.min(data)
        data /= np.max(data)
        return data

    @staticmethod
    def check_normality_shapiro(data: pd.DataFrame) -> np.ndarray:
        _, k = data.shape
        result = [None for _ in range(k)]
        for i in range(k):
            _, p_value = scipy.stats.shapiro(data[i])
            result[i] = p_value
        return np.array(result)

    @staticmethod
    def check_dispersion_equality(data: pd.DataFrame) -> float:
        statistic, p_value = scipy.stats.levene(*data.to_numpy().T)
        return p_value

    @staticmethod
    def apply_box_cox(data: pd.DataFrame) -> pd.DataFrame:
        _, k = data.shape
        result = data.copy()
        alphas = [0 for _ in range(k)]
        for i in range(k):
            result[i], alphas[i] = scipy.stats.boxcox(data[i])
        return result, alphas

    @staticmethod
    def fisher_test(data: pd.DataFrame, significance: float) -> bool:
        n, k = data.shape
        f_val = scipy.stats.f.ppf(q=1-significance, dfn=k-1, dfd=k*(n-1))

        sum_1 = 0
        sum_2 = 0
        for i in range(k):
            buff_2 = 0
            for j in range(n):
                sum_1 += data[i][j] ** 2
                buff_2 += data[i][j]
            sum_2 += buff_2 ** 2
        s_zero = (sum_1 - sum_2 / n) / (k * (n - 1))

        buff = 0
        mean_array = data.mean()
        global_mean = mean_array.mean()
        for i in range(k):
            buff += (mean_array[i] - global_mean) ** 2
        s_a_ = n * buff / (k - 1)

        return (s_a_ / s_zero) > f_val

    @staticmethod
    def kruskal_test(data: pd.DataFrame) -> float:
        statistic, p_value = scipy.stats.kruskal(*data.to_numpy().T)
        return p_value

    @staticmethod
    def welch_test(data) -> np.ndarray:
        k, _ = data.shape
        result = list()
        for i in range(k - 1):
            statistic, p_value = scipy.stats.ttest_ind(data[i], data[i + 1], equal_var=True)
            result.append(p_value)

        return np.array(result)

    @staticmethod
    def get_two_factor_data(data: pd.DataFrame, m: int = 5):
        n, k = data.shape
        new_n = n // m
        new_data = np.zeros((k, m, new_n))
        for i in range(k):
            col_name = data.columns[i]
            for j in range(m):
                for t in range(new_n):
                    new_data[i][j][t] = data[col_name][j * new_n + t]

        return new_data


def task_1(structure: DataStructure):
    for col in structure.df.columns:
        x = structure.df[col]
        fig, ax = plt.subplots(figsize=(12, 2.8), layout='constrained')
        ax.plot(x)
        ax.set_xlabel('Time')
        ax.set_ylabel('Voltage')
        ax.set_title(f"Parameter {col + 1}")
        plt.show()


def task_2(structure: DataStructure):
    idx_head = 'Param.'
    bins_number = 25
    significance = 0.05

    mean = structure.get_mean()
    dispersion = structure.get_dispersion()
    mode = pd.Series([structure.get_mode()[col].dropna().to_numpy() for col in structure.df.columns])
    median = structure.get_median()
    skewness = structure.get_skewness()
    kurtosis = structure.get_kurtosis()

    hypotheses_bool = list()
    for i in range(structure.k):
        hypotheses_bool.append(
            check_distribution_hypothesis(
                data=structure.df[i].to_numpy(),
                n=structure.n,
                interval_number=bins_number,
                math_expectation=mean[i],
                dispersion=dispersion[i],
                significance=significance
            )
        )

    hyp_bool_dict = {x[0]: [] for x in hypotheses_bool[0]}
    for el in hypotheses_bool:
        for key, val in el:
            hyp_bool_dict[key].append(val)

    print('-== Task 2 ==-\n')
    print_table([mean, dispersion], names=['Mean', 'Dispersion'], idx_head=idx_head)
    print_table([mode], names=['Mode'], cell_size=60, multi_val=True, idx_head=idx_head)
    print_table([median, skewness, kurtosis], names=['Median', 'Skewness', 'Kurtosis'], idx_head=idx_head)
    structure.get_histogram(bins_number=bins_number)
    print(f'Checking distribution hypothesis at significance {significance}:')
    print_table(list(hyp_bool_dict.values()), names=list(hyp_bool_dict.keys()), idx_head=idx_head, cell_size=12)


def task_3(structure: DataStructure):
    idx_head = 'Param.'
    significance = 0.05

    data_01 = structure.zero_one_transformation()
    shapiro_bool = structure.check_normality_shapiro(data_01) > significance
    dispersion_eq_bool = structure.check_dispersion_equality(data_01) > significance

    data_01_bc, alphas = structure.apply_box_cox(data_01)
    shapiro_bool_bc = structure.check_normality_shapiro(data_01) > significance
    dispersion_eq_bool_bc = structure.check_dispersion_equality(data_01) > significance

    normality_dis_hyp = shapiro_bool.all() or shapiro_bool_bc.all()
    dispersion_eq_hyp = dispersion_eq_bool or dispersion_eq_bool_bc

    if normality_dis_hyp and dispersion_eq_hyp:
        method_name = 'Fisher'
        res_bool = structure.fisher_test(data_01_bc, significance=significance)
    elif not normality_dis_hyp and not dispersion_eq_hyp:
        method_name = 'Kruskal'
        res_bool = structure.kruskal_test(data_01_bc) > significance
    else:
        method_name = 'Welch'
        res_bool = structure.welch_test(data_01_bc).all()

    print('-== Task 3 ==-\n')
    print(f'Result of Shapiro normality test at significance `{significance}`:')
    print_table([shapiro_bool], names=['Shapiro'], idx_head=idx_head, cell_size=10)
    print(f'Levene dispersion equality at significance `{significance}`: {dispersion_eq_bool}\n')

    print('--- Result after applying Box-Cox transformation ---')
    print(f'Result of Shapiro normality test at significance `{significance}`:')
    print_table([shapiro_bool_bc, alphas], names=['Shapiro', 'Cff. Alpha'], idx_head=idx_head, cell_size=10)
    print(f'Levene dispersion equality at significance `{significance}`: {dispersion_eq_bool_bc}\n')

    print(f'Result of 1-factor analyse ({method_name} method):')
    print(f'  Parameters influence equality: {res_bool}\n')


def task_4(structure: DataStructure):
    def get_mean_tf(data: np.ndarray):
        k_, m_ = len(data), len(data[0])
        result = np.zeros((k_, m_))
        for i in range(k_):
            for j in range(m_):
                result[i][j] = np.mean(data[i][j])
        return result

    def get_matrix_shapiro(tf_data, lvl_sgn):
        a_len, b_len, _ = tf_data.shape
        shapiro_matrix = np.zeros((a_len, b_len))
        for i in range(a_len):
            shapiro_matrix[i] = structure.check_normality_shapiro(tf_data[i].T) > lvl_sgn
        return shapiro_matrix == 1

    def get_box_cox_matrix(tf_data):
        a_len, b_len, x_len = tf_data.shape
        box_cox_matrix = np.zeros((a_len, b_len, x_len))
        alpha_matrix = np.zeros((a_len, b_len))
        for i in range(a_len):
            box_cox_res, alpha_res = structure.apply_box_cox(tf_data[i].T)
            box_cox_matrix[i], alpha_matrix[i] = box_cox_res.T, alpha_res
        return box_cox_matrix, alpha_matrix

    def standard_method(tf_mean_data, sgn_lvl):
        k, m = tf_mean_data.shape
        x_col_sum = np.sum(tf_mean_data, axis=0)
        x_row_sum = np.sum(tf_mean_data, axis=1)

        q_1 = np.sum(np.sum(np.square(tf_mean_data), axis=1))
        q_2 = np.sum(np.square(x_row_sum)) / m
        q_3 = np.sum(np.square(x_col_sum)) / k
        q_4 = np.square(np.sum(x_row_sum)) / (m * k)

        s_0 = (q_1 + q_4 - q_2 - q_3) / ((k - 1) * (m - 1))
        s_a = (q_2 - q_4) / (k - 1)
        s_b = (q_3 - q_4) / (m - 1)

        f_val_a = scipy.stats.f.ppf(q=1 - sgn_lvl, dfn=k - 1, dfd=(k - 1) * (m - 1))
        f_val_b = scipy.stats.f.ppf(q=1 - sgn_lvl, dfn=m - 1, dfd=(k - 1) * (m - 1))

        return (s_a / s_0) > f_val_a, (s_b / s_0) > f_val_b  # A_bool, B_bool

    def friedman_test_method(tf_mean_data, sgn_lvl):
        _, a_p_value = scipy.stats.friedmanchisquare(*tf_mean_data)
        _, b_p_value = scipy.stats.friedmanchisquare(*tf_mean_data.T)

        return a_p_value > sgn_lvl, b_p_value > sgn_lvl  # A_bool, B_bool

    significance = 0.05

    data_01 = structure.zero_one_transformation()
    two_factor_data = structure.get_two_factor_data(data_01)

    a_count, b_count, x_count = two_factor_data.shape
    a_names = [f'A{x + 1}' for x in range(a_count)]

    shapiro_bool_matrix = get_matrix_shapiro(tf_data=two_factor_data, lvl_sgn=significance)

    two_factor_data_bc, tf_bc_alpha = get_box_cox_matrix(tf_data=two_factor_data)
    shapiro_bool_matrix_bc = get_matrix_shapiro(tf_data=two_factor_data_bc, lvl_sgn=significance)

    normality_dis_hyp = shapiro_bool_matrix.all() or shapiro_bool_matrix_bc.all()
    two_factor_mean_bc = get_mean_tf(two_factor_data_bc)

    if normality_dis_hyp:
        method_name = 'Standard'
        a_bool, b_bool = standard_method(two_factor_mean_bc, sgn_lvl=significance)
    else:
        method_name = 'Friedman'
        a_bool, b_bool = friedman_test_method(two_factor_mean_bc, sgn_lvl=significance)

    print('-== Task 4 ==-\n')

    print(f'Result of Shapiro normality test at significance `{significance}`:')
    print_table(shapiro_bool_matrix, names=a_names, idx_head='', idx_prefix='B', idx_size=4, cell_size=5)

    print('--- Result after applying Box-Cox transformation ---')
    print(f'Result of coefficients `alpha` used for transformation:')
    print_table(tf_bc_alpha, names=a_names, idx_head='', idx_prefix='B', idx_size=4, cell_size=5)
    print(f'Result of Shapiro normality test at significance `{significance}`:')
    print_table(shapiro_bool_matrix_bc, names=a_names, idx_head='', idx_prefix='B', idx_size=4, cell_size=5)

    print(f'Result of 2-factor analyse ({method_name} method):')
    print(f'    Factor A influence equality: {a_bool}')
    print(f'    Factor B influence equality: {b_bool}\n')


def main():
    structure = DataStructure(filepath=FILE_PATH)

    task_1(structure)
    task_2(structure)
    task_3(structure)
    task_4(structure)


if __name__ == '__main__':
    main()
