# GI Researcher Tracker — Enriched Researcher List
# Updated with GitHub handles and ArXiv author query strings
# arxiv_query: used in ArXiv API call as au:<value>
# github: GitHub username, or None if not publicly active
# institution: current primary affiliation

RESEARCHERS = [
    {
        "name": "Yann LeCun",
        "twitter": "ylecun",
        "github": "ylecun",
        "arxiv_query": "LeCun_Y",
        "focus": "Deep Learning, Computer Vision",
        "institution": "Meta AI / NYU"
    },
    {
        "name": "Andrej Karpathy",
        "twitter": "karpathy",
        "github": "karpathy",
        "arxiv_query": "Karpathy_A",
        "focus": "Deep Learning, LLMs, Autonomous Driving",
        "institution": "Independent (ex-OpenAI, Tesla)"
    },
    {
        "name": "Fei-Fei Li",
        "twitter": "drfeifei",
        "github": None,
        "arxiv_query": "Li_Fei-Fei",
        "focus": "Computer Vision, Spatial AI, Human-Centered AI",
        "institution": "Stanford / World Labs"
    },
    {
        "name": "Demis Hassabis",
        "twitter": "demishassabis",
        "github": None,
        "arxiv_query": "Hassabis_D",
        "focus": "AGI, World Models, Game AI",
        "institution": "Google DeepMind"
    },
    {
        "name": "Jeff Dean",
        "twitter": "jeffdean",
        "github": None,
        "arxiv_query": "Dean_J",
        "focus": "Large Scale ML, Distributed Systems",
        "institution": "Google DeepMind"
    },
    {
        "name": "Pieter Abbeel",
        "twitter": "pabbeel",
        "github": "pabbeel",
        "arxiv_query": "Abbeel_P",
        "focus": "Robotics, Reinforcement Learning",
        "institution": "UC Berkeley / Covariant"
    },
    {
        "name": "Chelsea Finn",
        "twitter": "chelseabfinn",
        "github": "cbfinn",
        "arxiv_query": "Finn_C",
        "focus": "Meta-Learning, Robotics",
        "institution": "Stanford"
    },
    {
        "name": "Sergey Levine",
        "twitter": "svlevine",
        "github": None,
        "arxiv_query": "Levine_S",
        "focus": "Robotics, Deep Reinforcement Learning",
        "institution": "UC Berkeley"
    },
    {
        "name": "Ilya Sutskever",
        "twitter": "ilyasut",
        "github": None,
        "arxiv_query": "Sutskever_I",
        "focus": "Deep Learning, AGI, Safety",
        "institution": "SSI (ex-OpenAI)"
    },
    {
        "name": "Joscha Bach",
        "twitter": "joscha_bach",
        "github": "joschabach",
        "arxiv_query": "Bach_J",
        "focus": "Cognitive AI, Consciousness, World Models",
        "institution": "Independent / Harvard"
    },
    {
        "name": "Jim Fan",
        "twitter": "drjimfan",
        "github": "DrJimFan",
        # NOTE: His real name is Linxi "Jim" Fan — use this for ArXiv
        "arxiv_query": "Fan_Linxi",
        "focus": "Embodied AI, Foundation Models, Open-World Agents",
        "institution": "NVIDIA"
    },
    {
        "name": "Emad Mostaque",
        "twitter": "emostaque",
        "github": "emostaque",
        "arxiv_query": "Mostaque_E",
        "focus": "Generative AI, Open Source AI",
        "institution": "Independent (ex-Stability AI)"
    },
    {
        "name": "Francois Chollet",
        "twitter": "fchollet",
        "github": "fchollet",
        "arxiv_query": "Chollet_F",
        "focus": "Deep Learning, ARC, AI Benchmarks, Abstraction",
        "institution": "Google / ARC Prize"
    },
    {
        "name": "Lex Fridman",
        "twitter": "lexfridman",
        "github": "lexfridman",
        "arxiv_query": "Fridman_L",
        "focus": "Autonomous Vehicles, Human-Robot Interaction",
        "institution": "MIT / Independent"
    },
    {
        "name": "Gary Marcus",
        "twitter": "garymarcus",
        "github": None,
        "arxiv_query": "Marcus_G",
        "focus": "Cognitive Science, AI Critique, Hybrid AI",
        "institution": "NYU Emeritus / Independent"
    },
    {
        "name": "Kyunghyun Cho",
        "twitter": "kchonyc",
        "github": "kyunghyuncho",
        "arxiv_query": "Cho_K",
        "focus": "NLP, Generative Models, Seq2Seq",
        "institution": "NYU / Prescient Design"
    },
    {
        "name": "Oriol Vinyals",
        "twitter": "oriolvinyals",
        "github": None,
        "arxiv_query": "Vinyals_O",
        "focus": "Deep Learning, Game AI, AlphaStar",
        "institution": "Google DeepMind"
    },
    {
        "name": "David Silver",
        "twitter": "davidsilvermind",
        "github": None,
        "arxiv_query": "Silver_D",
        "focus": "Reinforcement Learning, AlphaGo, AlphaZero",
        "institution": "Google DeepMind"
    },
    {
        "name": "Abeba Birhane",
        "twitter": "Abebab",
        "github": "abeba",
        "arxiv_query": "Birhane_A",
        "focus": "AI Ethics, Cognitive Science, Critical AI Studies",
        "institution": "Trinity College Dublin / Mozilla"
    },
    {
        "name": "Percy Liang",
        "twitter": "percyliang",
        "github": "percyliang",
        "arxiv_query": "Liang_P",
        "focus": "Foundation Models, NLP, HELM Benchmarks",
        "institution": "Stanford / Together AI"
    },
]
