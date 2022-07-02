DIABETIC_MAPPING = {
    'Yes': 1,
    'No': 0,
    'No, borderline diabetes': 0.5,
    'Yes (during pregnancy)': 0.5
}
GENERAL_HEALTH_MAPPING = {
    'Very good': 4,
    'Good': 3,
    'Excellent': 5,
    'Fair': 2,
    'Poor': 1,
}


def transform_age(age: str) -> float:
    """
    convert age interval into it's mean
    :param age:
    :return: mean age
    """
    if age == '80 or older':
        return 80
    start_age = int(age.split('-')[0])
    end_age = int(age.split('-')[1])
    return (start_age + end_age) / 2


def transform_diabetic(diabetic: str) -> float:
    """
    transform diabet feature into ordinal
    :param diabetic: str
    :return: float ordinal value
    """
    return DIABETIC_MAPPING[diabetic]


def transform_general_health(gen_health: str) -> int:
    """
    transform general heath into ordinal values
    :param gen_health:
    :return:
    """
    return GENERAL_HEALTH_MAPPING[gen_health]

