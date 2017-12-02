from unittest import TestCase, main, skip
from snippets import Snippet


class TestSnippets(TestCase):
    def test_hide_target_phrase(self):
        text = """Wpłynąłem na suchego przestwór oceanu,
Wóz nurza się w zieloność i jak łódka brodzi,
Śród fali łąk szumiących, śród kwiatów powodzi,
Omijam koralowe ostrowy burzanu.""".replace("\n", " ")
        s1 = Snippet(text, "przestwór oceanu")
        self.assertEqual(s1.target_indices, [3])
        self.assertEqual(s1.masked_snippet, ['wpłynąłem', 'na', 'suchego', '<UNKNOWN>', ',', 'wóz', 'nurza', 'się', 'w', 'zieloność', 'i', 'jak', 'łódka', 'brodzi', ',', 'śród', 'fali', 'łąk', 'szumiących', ',', 'śród', 'kwiatów', 'powodzi', ',', 'omijam', 'koralowe', 'ostrowy', 'burzanu', '.'])

    def test_hide_tartget_phrase_multiple(self):
        text1 = "a b c d a b a a b b a"
        s1 = Snippet(text1, "b")
        self.assertEqual(
            s1.masked_snippet,
            ["a", "<UNKNOWN>", "c", "d", "a", "<UNKNOWN>", "a", "a", "<UNKNOWN>", "<UNKNOWN>", "a"])
        self.assertEqual(s1.target_indices, [1, 5, 8, 9])
        s2 = Snippet(text1, "b b")
        self.assertEqual(
            s2.masked_snippet,
            ["a", "b", "c", "d", "a", "b", "a", "a", "<UNKNOWN>", "a"]
        )
        self.assertEqual(s2.target_indices, [8])

    def test_hide_tartget_phrase_edges(self):
        text1 = "a d c a"
        s1 = Snippet(text1, "a")
        self.assertEqual(s1.masked_snippet, ["<UNKNOWN>", "d", "c", "<UNKNOWN>"])
        self.assertEqual(s1.target_indices, [0, 3])

if __name__ == "__main__":
    main()
