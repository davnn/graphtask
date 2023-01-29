from types import MappingProxyType as MapProxy

from hypothesis import strategies as s

max_size = 3

# values of basic types
text = s.text()
binary = s.binary()
ints = s.integers()
floats = s.floats(allow_nan=False)
complexes = s.complex_numbers(allow_nan=False)
boolean = s.booleans()
nones = s.none()
basics = s.one_of([text, binary, ints, floats, complexes, boolean, nones])

# sequences of basic types
create_list = lambda strategy: s.lists(strategy, max_size=max_size)
create_set = lambda strategy: s.sets(strategy, max_size=max_size)
create_tuple = lambda strategy: s.one_of([s.tuples(*([strategy] * i)) for i in range(max_size)])
create_dict = lambda strategy: s.dictionaries(keys=text, values=strategy, max_size=max_size)
create_nonempty_dict = lambda strategy: s.dictionaries(keys=text, values=strategy, max_size=max_size, min_size=1)
create_immutable_mapping = lambda strategy: s.builds(lambda d: MapProxy(d), create_dict(strategy))
basic_containers = [create_list(basics), create_set(basics), create_tuple(basics), create_dict(basics)]
lists, sets, tuples, dicts = basic_containers
sequences = s.one_of(basic_containers)  # these all have a length defined

# iterables of basic types
create_iterable = lambda strategy: s.iterables(strategy, max_size=max_size)
iterables = create_iterable(basics)

# any value is valid
anything = s.one_of([basics, sequences, iterables])

# context dependant strategies
int_gt_1_lt_max = s.integers(min_value=2, max_value=max_size)
list_of_iterables = create_list(iterables)
dict_of_iterables = create_dict(iterables)
