import math
from numpy import exp as np_exp
from scipy.stats import chi2


def print_table(table_list,
                names: list,
                idx_size: int = 6,
                idx_head: str = 'iter',
                idx_prefix: str = '',
                cell_size: int = 16,
                multi_val: bool = False):
    def set_centered(data: str, max_size: int):
        return ' ' * ((max_size - len(data)) // 2) + data

    def get_list_representation(data: list, max_size: int):
        brackets = ('[', ']')
        is_started = False

        result = brackets[0]
        last_element = '...'

        last_el_len = len(last_element)
        counter = 1
        for e in data:
            if is_started:
                result += ', '
            else:
                is_started = True

            str_el = str(e)
            add_len = len(str_el) + 2
            if counter + add_len + last_el_len <= max_size:
                result += f'{str_el}'
                counter += add_len

            elif counter + last_el_len <= max_size:
                result += f'{last_element}'
                counter += last_el_len
                return result

            else:
                break

        return result + brackets[-1]

    output_list = list()
    width = len(names)
    height = len(table_list[0])

    idx_template = '{:' + str(idx_size) + '}'
    cell_template = '{:' + str(cell_size) + '}'

    single_v_line = '-' * (width * (cell_size + 3) + idx_size + 4)
    output_list.append(single_v_line)

    header_str = f'| {idx_template.format(set_centered(data=idx_head, max_size=idx_size))} |'
    for i in range(width):
        value = names[i]
        header_str += f' {cell_template.format(set_centered(data=value, max_size=cell_size))} |'
    output_list.append(header_str)
    output_list.append(single_v_line)

    for i in range(height):
        row_str = f'| {idx_template.format(idx_prefix + str(i + 1))} |'
        for j in range(width):
            cell_content = table_list[j][i]
            if multi_val:
                val = get_list_representation(cell_content, max_size=cell_size - 2)
                row_str += f' {cell_template.format(val)} |'
            else:
                round_number = cell_size - 1 - len(str(cell_content).split('.')[0])
                try:
                    value = round(cell_content, round_number if round_number > 0 else 0)
                except TypeError:
                    value = str(cell_content)
                row_str += f' {cell_template.format(value)} |'

        output_list.append(row_str)
        output_list.append(single_v_line)

    for el in output_list:
        print(el)
    print()


def get_intervals(data: list, intervals_number: int, epsilon: float = 0.001):
    min_val, max_val = min(data), max(data)
    step = (max_val - min_val) / intervals_number
    result = [min_val + x * step for x in range(intervals_number)]
    result.append(max_val * (1 + epsilon))
    return result


def get_frequency_list(data, intervals_number: int = 10):
    intervals_list = get_intervals(data=data, intervals_number=intervals_number)
    result = [0 for _ in range(intervals_number)]

    for i in range(intervals_number):
        count = 0
        start, end = intervals_list[i], intervals_list[i + 1]
        for el in data:
            if start <= el < end:
                count += 1
        result[i] = count
    return result


def frequency_table_print(header, data, spacing: int = 20, id_spacing: int = 10):
    ln = len(header) - 1
    template_left = "|{:" + str(id_spacing) + "} |"
    template_main = "{:" + str(spacing) + "} |"
    vertical_line = '-' * (ln * (spacing + 2) + id_spacing + 3)

    def top_part_print():
        print(vertical_line)

        title = header[0]
        h = template_left.format((' ' * int((id_spacing - len(title)) / 2)) + title)
        for el in header[1:]:
            h += template_main.format((' ' * int((spacing - len(el)) / 2)) + el)

        print(h)

    def body_part_print():
        for i in range(len(data)):
            print(vertical_line)

            sub_title = 'Delta ' + str(i + 1)
            s = template_left.format((' ' * int((id_spacing - len(sub_title)) / 2)) + sub_title)
            for el in data[i]:
                s += template_main.format(round(el, spacing - 4))

            print(s)

        print(vertical_line)

    def bottom_part_print():
        total_sum = sum([x[0] for x in data])
        sum_title = "Sum = " + str(total_sum)
        end_sum = template_left.format((' ' * id_spacing))
        end_sum += template_main.format((' ' * int((spacing - len(sum_title)) / 2)) + sum_title)

        for i in range(ln - 1):
            end_sum += template_main.format((' ' * spacing))

        print(end_sum)

    top_part_print()
    body_part_print()
    bottom_part_print()

    print(vertical_line, '\n')


def check_normal_distribution(data, n, math_expectation, dispersion, frequency_list, significance: float):
    def gaussian_function(math_expectation_, dispersion_):
        return lambda x: (np_exp(- pow((x - math_expectation_) / dispersion_, 2) / 2)) / pow(2 * math.pi, 1/2)

    def get_theoretical_frequency(math_expectation_, dispersion_, n_, intervals_):
        h = intervals_[2] - intervals_[1]
        phi = gaussian_function(math_expectation_=math_expectation_, dispersion_=dispersion_)

        result_ = [(n_ * h * phi(x + h/2)) / dispersion_ for x in intervals_[:-1]]
        return result_

    intervals_number = len(frequency_list)
    intervals = get_intervals(data=data, intervals_number=intervals_number)
    theoretical_frequency_list = get_theoretical_frequency(
        math_expectation_=math_expectation,
        dispersion_=dispersion,
        n_=n,
        intervals_=intervals
    )

    epsilon = 1e-4
    chi_observed = 0
    for j in range(intervals_number):
        n_i = frequency_list[j]
        n_dash_i = theoretical_frequency_list[j]
        if n_dash_i > epsilon:
            chi_observed += pow(n_i - n_dash_i, 2) / n_dash_i
        else:
            chi_observed += pow(n_i, 2) / epsilon

    chi_critical = chi2.ppf(1 - significance, intervals_number - 3)
    result = chi_critical > chi_observed
    return result


def check_exponential_distribution(data, n, math_expectation, frequency_list, significance: float):
    def get_theoretical_frequency(math_expectation_, n_, intervals_):
        epsilon_ = 1e-2
        signum = 1. if math_expectation_ >= 0 else -1.
        if abs(math_expectation_) > epsilon_:
            lambda_parameter = 1 / math_expectation_
        else:
            lambda_parameter = 1 / signum * epsilon_

        intervals_count_ = len(intervals_) - 1

        result_ = [0 for _ in range(intervals_count_)]
        for i in range(intervals_count_):
            p_i = np_exp(- lambda_parameter * intervals_[i]) - np_exp(- lambda_parameter * intervals_[i + 1])
            result_[i] = n_ * p_i
        return result_

    intervals_number = len(frequency_list)
    intervals = get_intervals(data=data, intervals_number=intervals_number)
    theoretical_frequency_list = get_theoretical_frequency(
        math_expectation_=math_expectation,
        n_=n,
        intervals_=intervals
    )

    epsilon = 1e-4
    chi_observed = 0
    for j in range(intervals_number):
        n_i = frequency_list[j]
        n_dash_i = theoretical_frequency_list[j]
        if n_dash_i > epsilon:
            chi_observed += pow(n_i - n_dash_i, 2) / n_dash_i
        else:
            chi_observed += pow(n_i, 2) / epsilon

    chi_critical = chi2.ppf(1 - significance, intervals_number - 2)
    result = chi_critical > chi_observed
    return result


def check_uniform_distribution(data, n, math_expectation, dispersion, frequency_list, significance: float):
    def get_theoretical_frequency(n_, density_, a_, b_, intervals_):
        intervals_count_ = len(intervals_) - 1
        result_ = [0 for _ in range(intervals_count_)]
        for i in range(intervals_count_):
            if i == 0:
                result_[i] = n_ * density_ * (intervals_[i + 1] - a_)
            elif i == intervals_count_ - 1:
                result_[i] = n_ * density_ * (b_ - intervals_[i])
            else:
                result_[i] = n_ * density_ * (intervals_[i + 1] - intervals_[i])
        return result_

    sqrt_3_sigma = pow(3 * dispersion, 1/2)
    a, b = math_expectation - sqrt_3_sigma, math_expectation + sqrt_3_sigma
    density = 1 / (b - a)

    intervals_number = len(frequency_list)
    intervals = get_intervals(data=data, intervals_number=intervals_number)
    theoretical_frequency_list = get_theoretical_frequency(n_=n, density_=density, a_=a, b_=b, intervals_=intervals)

    chi_observed = 0
    for j in range(intervals_number):
        n_i = frequency_list[j]
        n_dash_i = theoretical_frequency_list[j]
        chi_observed += pow(n_i - n_dash_i, 2) / n_dash_i
    chi_observed = abs(chi_observed)

    chi_critical = chi2.ppf(1 - significance, intervals_number - 3)
    result = chi_critical > chi_observed
    return result


def check_distribution_hypothesis(data, n, math_expectation, dispersion, interval_number, significance: float = 0.05):
    frequency_list = get_frequency_list(data=data, intervals_number=interval_number)
    result = tuple([
        ("Normal", check_normal_distribution(
            data=data,
            n=n,
            math_expectation=math_expectation,
            dispersion=dispersion,
            frequency_list=frequency_list,
            significance=significance
        )),
        ("Exponential", check_exponential_distribution(
            data=data,
            n=n,
            math_expectation=math_expectation,
            frequency_list=frequency_list,
            significance=significance
        )),
        ("Uniform", check_uniform_distribution(
            data=data,
            n=n,
            math_expectation=math_expectation,
            dispersion=dispersion,
            frequency_list=frequency_list,
            significance=significance
        ))
    ])
    return result
