from unittest import TestCase
from unittest import main as run_tests


from pypermute import Permutation


class NoError(BaseException):
    ...


class TestPermutation(TestCase):

    def test1(self):
        """Wrong representations
        """
        test_cases = [
            ('empty representation', (), ValueError),
            ('empty cycle', ((),), ValueError),
            ('wrong index type (float)', ((0.0,),), TypeError),
            ('missing index 0', ((1, 2), (4, 3)), ValueError),
            ('missing index', ((0, 1), (3,), (4,), ), ValueError),
            ('duplicate index', ((0,), (1,), (2, 3), (1,)), ValueError)
        ]

        for title, representation, expected_exception in test_cases:
            with self.subTest(msg=title):
                with self.assertRaises(expected_exception):
                    _ = Permutation(representation)

    def test2(self):
        """Valid representations, testing structural properties
        """
        test_cases = [
            ('valid transposition', ((0,), (1, 2,)), 1, 1),
            ('valid two cycles', ((0, 2,), (3, 1,)), 0, 2),
            ('valid identity', ((0,), (1,), (2,),), 3, 0),
        ]

        for title, representation, nb_fixed_points, nb_cycles in test_cases:
            with self.subTest(msg=title):
                p = Permutation(representation)
                self.assertEqual(p.fixed_points, nb_fixed_points)
                self.assertEqual(p.cycles, nb_cycles)

    def test3(self):
        """Functional testing"""
        t = None
        try:
            t = Permutation(((0, 2,), (1, 3,), (4, 5, 7), (6,),))
        except AssertionError:
            self.assertFalse(True)

        with self.subTest('cycles == 3'):
            self.assertEqual(t.cycles, 3)

        with self.subTest('fixed point == 1'):
            self.assertEqual(t.fixed_points, 1)

        with self.subTest('Two transpositions'):
            self.assertEqual(t.cycles_by_length(2), 2)

        with self.subTest('One cycle of length 3'):
            self.assertEqual(t.cycles_by_length(3), 1)

        with self.subTest('Actually permutes something'):
            s = 'A B C D E F G H'
            s = t.permute_sequence(list(s.split(' ')))
            self.assertEqual(' '.join(s), 'C D A B F H G E')

        with self.subTest('Transpositions are cycle of length 2'):
            self.assertTrue(t.transpositions == t.cycles_by_length(2))

    def test4(self):
        """Definition from a product of cycles"""

        with self.assertRaises(PermissionError):
            Permutation(reach=0).add_cycle(None)

        with Permutation(reach=3) as p:
            p.add_cycle(0, 1)
            p.add_cycle(2)

        with self.assertRaises(PermissionError):
            p.add_cycle(None)

        self.assertEqual(p.fixed_points, 1)
        self.assertEqual(p.cycles, 1)
        self.assertEqual(p.transpositions, 1)

    def test5(self):
        """Random full cycle
        """
        with self.assertRaises(NoError):
            p = Permutation.random_full_cycle(5)
            raise NoError

        self.assertTrue(p.reach == 5)
        self.assertEqual(p.fixed_points, 0)
        self.assertEqual(p.cycles, 1)

    def test6(self):
        """Identity
        """
        with self.assertRaises(NoError):
            p = Permutation.identity(5)
            raise NoError

        self.assertTrue(p.reach == 5)
        self.assertEqual(p.fixed_points, p.reach)
        self.assertEqual(p.cycles, 0)

    def test7(self):
        """Building context and add_cycle
        """
        with Permutation(reach=9) as p:
            p.add_cycle(1)
            p.add_cycle(4, 2)
            p.add_cycle(3, 5)
            p.add_cycle(8)
            p.add_cycle(6, 7, 0)

        r = p.canonical()
        self.assertEqual(r, ((8,), (7, 0, 6), (5, 3), (4, 2), (1,)))

    def test8(self):
        """Inverse permutation
        """
        rep = ((8,), (7, 0, 6), (5, 3), (4, 2), (1,))
        p = Permutation(rep)
        p_1 = ~p  # inverse permutation
        assert p_1 * p == Permutation.identity(p.reach)

    def test9(self):
        """Permutation from a bijective map
        """
        m = {0: 1, 1: 3, 2: 2, 3: 0}
        p = Permutation.from_map(m)

        self.assertEqual(p.fixed_points, 1)
        self.assertEqual(p.cycles, 1)
        self.assertEqual(p.canonical(), ((3, 0, 1), (2,)))

    def test90(self):
        """from_map to_map equivalence"""
        m = {0: 1, 1: 3, 2: 2, 3: 0}
        p = Permutation.from_map(m)

        self.assertEqual(p.to_map(), m)

    def test91(self):
        """Cyclotomic permutations for 3, 4, 5 and 6-th roots of 1
        """
        with self.subTest('cyclotomic odd w/ fixed pts'):
            p1 = Permutation.cyclotomic(3)
            self.assertEqual(p1.canonical(), ((2, 1), (0,)))

        with self.subTest('cyclotomic even w/ fixed pts'):
            p2 = Permutation.cyclotomic(4)
            self.assertEqual(p2.canonical(), ((3, 1), (2,), (0, )))

        with self.subTest('cyclotomic odd wo/ fixed pts'):
            p3 = Permutation.cyclotomic(5, allows_fixed_points=False)
            self.assertEqual(p3.canonical(), ((3, 0), (2, 1)))

        with self.subTest('cyclotomic even wo/ fixed pts'):
            p4 = Permutation.cyclotomic(6, allows_fixed_points=False)
            self.assertEqual(p4.canonical(), ((5, 0, 1, 2, 3, 4,),))

    def test92(self):
        """Cyclotomic permutation for 17 vertices
        """
        p = Permutation.cyclotomic(17, allows_fixed_points=False)
        for cycle in p:
            self.assertEqual(len(cycle), 2)

    def test93(self):
        """Cyclotomic conjugation for 5 vertices
        """
        c = Permutation.cyclotomic(5, allows_fixed_points=False)
        self.assertTrue(c ** 2 == c.identity(4))

    def test94(self):
        """Circular permutation
        """
        c = Permutation.circular(5)
        self.assertEqual(c.canonical(), ((4, 0, 1, 2, 3,),))

    def test95(self):
        """Idempotence of circular and reversed circular permutations
        """
        c1 = Permutation.circular(5)
        c2 = Permutation.circular(5, reverse=True)

        self.assertTrue(c1 * c2 == c2 * c1 == c1.identity(5))

    def test96(self):
        """Full random permutation sanity check"""
        with self.assertRaises(NoError):
            p = Permutation.random(5)
            raise NoError

    def test97(self):
        """Full random permutation w/o fixed points"""
        p = Permutation.random(5, allows_fixed_points=False)
        self.assertEqual(p.fixed_points, 0)


if __name__ == '__main__':
    run_tests()
