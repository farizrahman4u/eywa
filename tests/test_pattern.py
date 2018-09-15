from eywa.nlu import Pattern


class TestPattern:

    def test_pattern(self):

        p = Pattern('[fruit : apple, banana] is my favourite fruit')
        assert p('i like grapes') == {'fruit': 'grapes'}

    def test_pattern_serialization(self):

        p1 = Pattern('[fruit : apple, banana] is my favourite fruit')
        p1('i like grapes')

        config = p1.serialize()
        p2 = Pattern.deserialize(config)

        test_inputs = ['i like grapes']

        for test_input in test_inputs:
            p1_out = p1(test_input)
            p2_out = p2(test_input)
            assert p1_out == p2_out
