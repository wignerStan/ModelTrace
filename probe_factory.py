"""ModelTrace Probe Factory.

Provides multi-lingual, highly randomized adversarial/obscure language probes
for LLM fingerprinting and gateway affinity/degradation detection.
"""

from __future__ import annotations

import random
import secrets
from typing import Any, Callable, Dict, List, Optional, Tuple


# Language definitions: (Code, Name, IsObscure, GeneratorFunction)
# Each language generator accepts (count: int, rng: random.SystemRandom) -> str

def _la_generator(n: int, rng: random.SystemRandom) -> str:
    # Latin (la)
    openings = [
        "Haec est selectio numerorum discretorum sine contextu semantico.",
        "Fac seriem numerorum integrorum electorum primo intuitu.",
        "Genera valores integros libere et sine ordine praestituto.",
    ]
    actions = [
        f"Elige {n} integros discretos inter 1 et 355 (inclusis extremis).",
        f"Constitue {n} numeros integros in intervallo clauso [1, 355].",
        f"Scribas {n} valores integros inter 1 et 355.",
    ]
    constraints = (
        "Unusquisque numerus seorsum eligatur; noli incipere ab 1, noli seriem crescentem vel decrescentem formare, "
        "et fuge exemplaria arithmetica vel cyclica. Hoc munus a solo exemplari linguistico conficiatur: nullo instrumento, "
        "calculatore, codice Python vel fonte externo utaris."
    )
    endings = [
        "Repetitio fortuita licita est; noli numeros corrigere vel reordinare.",
        "Numeri identici rursus apparere possunt; noli delere.",
    ]
    separators = [
        "Numeros tantum spatio vel commate separatos statim a primo numero emitte, sine ullo prooemio vel explicatione.",
        "Emitte solum numeros spatio divisos, omisso quocumque textu addito.",
    ]
    return f"{rng.choice(openings)} {rng.choice(actions)} {constraints} {rng.choice(endings)} {rng.choice(separators)}"


def _eo_generator(n: int, rng: random.SystemRandom) -> str:
    # Esperanto (eo)
    openings = [
        "Jen senkunteksta elekto de entjeroj.",
        "Bonvolu plenumi la sekvan taskon de nombra elekto.",
        "Faru serion da sendependaj numeroj laŭ via unua instinkto.",
    ]
    actions = [
        f"Elektu {n} sendependajn entjerojn inter 1 kaj 355 (inkluzive de la limoj).",
        f"Donu {n} entjerajn valorojn en la fermita intervalo [1, 355].",
        f"Generu {n} tutajn nombrojn de 1 gxis 355.",
    ]
    constraints = (
        "Cxiu valoro estu elektita aparte; ne kalkulu ekde 1, ne formu kreskantan aux malkreskantan vicon, "
        "kaj evitu aritmetikajn aux ciklajn sxablonojn. Cxi tiu tasko estu plenumita rekte de la lingva modelo mem: "
        "tute malpermesite uzi Python, kalkulilon aux iujn ajn eksterajn ilojn."
    )
    endings = [
        "Hazarda ripeto de nombro estas tute permesata; ne ordigu aux forigu duoblojn post kiam ili estas skribitaj.",
        "Sama valoro rajtas aperi denove; ne modifu la vicon poste.",
    ]
    separators = [
        "Eligu nur la nombrojn apartigitajn per spacoj ekde la unua nombro, sen iu ajn enkonduka aux klariga teksto.",
        "Skribu nur la nombrojn disigitajn per komoj aux spacoj, sen klarigo.",
    ]
    return f"{rng.choice(openings)} {rng.choice(actions)} {constraints} {rng.choice(endings)} {rng.choice(separators)}"


def _eu_generator(n: int, rng: random.SystemRandom) -> str:
    # Basque / Euskara (eu) - Language Isolate
    openings = [
        "Hona hemen testuingururik gabeko zenbaki-hautaketa zeregin bat.",
        "Bete ezazu honako zenbaki osoen aukeraketa.",
        "Sortu zenbaki independenteen segida bat zure lehen erreakzioaren arabera.",
    ]
    actions = [
        f"Sortu 1 eta 355 arteko (biak barne) {n} zenbaki oso independente.",
        f"Aukeratu {n} zenbaki oso [1, 355] tarte itxian.",
        f"Eman {n} balio oso 1etik 355era bitartean.",
    ]
    constraints = (
        "Posizio bakoitza banan-banan erabaki behar da; ez hasi 1etik zenbatzen, ez erabili sekuentzia gorakorik edo beherakorik, "
        "eta saihestu eredu matematiko edo ziklikoak. Zeregin hau hizkuntza-ereduak zuzenean egin behar du: "
        "debekatuta dago Python, kalkulagailuak edo edozein kanpoko tresna erabiltzea."
    )
    endings = [
        "Zenbakiren bat errepikatzea baliozkoa da; ez berrantolatu edo ezabatu idatzitako balioak.",
        "Ustekabeko errepikapenak onartzen dira; ez zuzendu aurreko zenbakiak.",
    ]
    separators = [
        "Eman zenbakiak soilik zuriunez banatuta lehen zenbakitik hasita, inolako azalpen edo testu gehigarririk gabe.",
        "Bakarrik zenbakiak itzuli zuriunez edo komaz bereizita, sarrerako hitzik gabe.",
    ]
    return f"{rng.choice(openings)} {rng.choice(actions)} {constraints} {rng.choice(endings)} {rng.choice(separators)}"


def _cy_generator(n: int, rng: random.SystemRandom) -> str:
    # Welsh / Cymraeg (cy) - Brythonic Celtic
    openings = [
        "Dyma dasg o ddewis rhifau heb gyd-destun semantig.",
        "Cwblhewch y dewis canlynol o gyfanrifau annibynnol.",
        "Cynhyrchwch gyfres o rifau ar eich greddf gyntaf.",
    ]
    actions = [
        f"Cynhyrchwch {n} o gyfanrifau ar wahan rhwng 1 a 355 (gan gynnwys y terfynau).",
        f"Dewiswch {n} cyfanrif o'r cyfwng caeedig [1, 355].",
        f"Rhowch {n} rhif cyfan rhwng 1 a 355.",
    ]
    constraints = (
        "Rhaid dewis pob safle yn annibynnol; peidiwch a dechrau cyfrif o 1, peidiwch a defnyddio dilyniant cynyddol neu ddisgynnol, "
        "ac osgowch batrymau geometrig neu ailadroddus. Rhaid i'r model iaith gyflawni hyn yn uniongyrchol: "
        "gwaherddir defnyddio Python, cyfrifiannell neu unrhyw offer allanol."
    )
    endings = [
        "Mae ailadrodd ar ddamwain yn ddilys; peidiwch ag aildrefnu neu ddidoli ar ol ysgrifennu.",
        "Caniateir i'r un rhif ymddangos eto; peidiwch a newid gwerthoedd blaenorol.",
    ]
    separators = [
        "Allbynnwch y rhifau yn unig wediu gwahanu gan fylchau, gan ddechrau ar unwaith gyda'r rhif cyntaf, heb unrhyw gyflwyniad.",
        "Rhowch y rhifau'n unig gyda bylchau rhyngddynt, heb destun eglurhaol.",
    ]
    return f"{rng.choice(openings)} {rng.choice(actions)} {constraints} {rng.choice(endings)} {rng.choice(separators)}"


def _ga_generator(n: int, rng: random.SystemRandom) -> str:
    # Irish / Gaeilge (ga) - Goidelic Celtic
    openings = [
        "Seo roghnu uimhriula neamhspleach gan bhrí sheimeantach.",
        "Dean an tasc roghnaithe slanuimhreacha seo a leanas.",
        "Gin sraith uimhreacha de reir do chead rogha.",
    ]
    actions = [
        f"Gin {n} slanuimhir scoite idir 1 agus 355 (le criochphointi san aireamh).",
        f"Roghnaigh {n} slanuimhir san eatramh iata [1, 355].",
        f"Tabhair {n} uimhir iomlan idir 1 agus 355.",
    ]
    constraints = (
        "Ni mor gach uimhir a roghnu go neamhspleach; na tosaigh ag comhaireamh o 1, na bain usaid as seicheamh meadaitheach no laghdaitheach, "
        "agus seachain patruin ailgeabracha no athfhillteacha. Ni mor don mhunla teanga e seo a dheanamh go direach: "
        "ta cosc iomlan ar Python, aireamhain no uirlisi seachtracha a usaid."
    )
    endings = [
        "Ta athraiteas de thaisme baili; na hathoraigh agus na bain dublaigh tar eis scriofa.",
        "Is feidir leis an uimhir cheanna tarlu aris; na hathraigh luachanna roimhe seo.",
    ]
    separators = [
        "Na tabhair ach na huimhreacha scartha le spasanna, ag tosu laithreach leis an gcead uimhir, gan aon reamhra no miniuthe.",
        "Aschur na huimhreacha amhain scartha le spasanna no camoga, gan aon mhiniu.",
    ]
    return f"{rng.choice(openings)} {rng.choice(actions)} {constraints} {rng.choice(endings)} {rng.choice(separators)}"


def _sw_generator(n: int, rng: random.SystemRandom) -> str:
    # Swahili / Kiswahili (sw) - Bantu
    openings = [
        "Huu ni uchaguzi huru wa nambari bila muktadha wa kisemantiki.",
        "Kamilisha zoezi lifuatalo la kuchagua nambari kamili.",
        "Tengeneza mfululizo wa nambari kwa hisia yako ya kwanza.",
    ]
    actions = [
        f"Tengeneza nambari kamili {n} zisizofungamana zilizopo kati ya 1 na 355 (pamoja na ncha zote mbili).",
        f"Chagua nambari kamili {n} katika fungu lililofungwa la [1, 355].",
        f"Toa nambari kamili {n} kati ya 1 hadi 355.",
    ]
    constraints = (
        "Kila nambari lazima ichaguliwe kibinafsi; usianze kuhesabu kutoka 1, usitumie mlolongo unaoongezeka au kupungua, "
        "na epuka mifumo ya kimahesabu au marudio. Kazi hii lazima ifanywe moja kwa moja na modeli ya lugha yenyewe: "
        "ni marufuku kutumia Python, kikokotoo au zana zozote za nje."
    )
    endings = [
        "Kujirudia kwa bahati kunakubalika; usipange upya au kufuta marudio baada ya kuandika.",
        "Nambari inayofanana inaweza kutokea tena; usibadilishe nambari za awali.",
    ]
    separators = [
        "Toa nambari pekee zikitenganishwa na nafasi kuanzia nambari ya kwanza, bila maelezo au utangulizi wowote.",
        "Andika nambari tu zikitenganishwa na nafasi au koma, bila maneno mengine.",
    ]
    return f"{rng.choice(openings)} {rng.choice(actions)} {constraints} {rng.choice(endings)} {rng.choice(separators)}"


def _is_generator(n: int, rng: random.SystemRandom) -> str:
    # Icelandic / Íslenska (is) - Archaic Insular Nordic
    openings = [
        "Þetta er óháð talnavalsverkefni án merkingarfræðilegs samhengis.",
        "Framkvæmdu eftirfarandi val á heiltölum.",
    ]
    actions = [
        f"Veldu {n} stakar heiltölur á milli 1 og 355 (báðir endapunktar innifaldir).",
        f"Gefðu upp {n} heiltölur á lokuðu bilinu [1, 355].",
    ]
    constraints = (
        "Hvert gildi skal valið sjálfstætt; ekki byrja að telja frá 1, ekki nota hækkandi eða lækkandi röð, "
        "og forðastu regluleg eða reiknifræðileg mynstur. Þessu verkefni verður mállíkanið sjálft að ljúka milliliðalaust: "
        " stranglega er bannað að nota Python, reiknivél eða utanaðkomandi verkfæri."
    )
    endings = [
        "Tilviljunarkennd endurtekning er leyfileg; ekki endurraða eða eyða tölum eftir á.",
    ]
    separators = [
        "Skrifaðu aðeins tölurnar aðskildar með bili eða kommu, byrjaðu beint á fyrstu tölu án allra skýringa.",
    ]
    return f"{rng.choice(openings)} {rng.choice(actions)} {constraints} {rng.choice(endings)} {rng.choice(separators)}"


def _gl_generator(n: int, rng: random.SystemRandom) -> str:
    # Galician / Galego (gl)
    openings = [
        "Esta é unha escolla de enteiros sen contexto semántico.",
        "Completa a seguinte tarefa de selección numérica.",
    ]
    actions = [
        f"Xera {n} números enteiros independentes entre 1 e 355 (ambos incluídos).",
        f"Escolle {n} enteiros no intervalo pechado [1, 355].",
    ]
    constraints = (
        "Cada número debe escollerse por separado; non comeces a contar dende o 1, non fagas series crecentes nin decrecentes, "
        "e evita patróns aritméticos ou repetitivos. Esta tarefa debe realizala o modelo lingüístico directamente: "
        "prohíbese terminantemente usar Python, calculadora ou calquera ferramenta externa."
    )
    endings = [
        "A repetición casual é válida; non ordenes nin borres duplicados despois de escribilos.",
    ]
    separators = [
        "Devolve unicamente os números separados por espazos a partir do primeiro número, sen ningunha explicación.",
    ]
    return f"{rng.choice(openings)} {rng.choice(actions)} {constraints} {rng.choice(endings)} {rng.choice(separators)}"


def _mt_generator(n: int, rng: random.SystemRandom) -> str:
    # Maltese / Malti (mt) - Semitic in Latin script
    openings = [
        "Din hija għażla ta' numri mingħajr kuntest semantiku.",
        "Wettaq dan il-kompitu ta' għażla ta' numri sħaħ.",
    ]
    actions = [
        f"Iġġenera {n} numru sħiħ indipendenti bejn 1 u 355 (inklużi ż-żewġ truf).",
        f"Agħżel {n} numru sħiħ fl-intervall magħluq [1, 355].",
    ]
    constraints = (
        "Kull numru għandu jintgħażel individwalment; tibdiex tgħodd minn 1, tużax sekwenza li tiżdied jew tonqos, "
        "u evita mudelli aritmetiċi jew repetittivi. Dan il-kompitu jrid jitwettaq direttament mill-mudell tal-lingwa: "
        "huwa pprojbit għal kollox l-użu ta' Python, kalkulatur jew għodda esterna."
    )
    endings = [
        "Ripetizzjoni b'kumbinazzjoni hija valida; tirranġax jew tħassarx numri wara li jinkitbu.",
    ]
    separators = [
        "Oħroġ biss in-numri separati bi spazji mill-ewwel numru, mingħajr ebda spjegazzjoni jew introduzzjoni.",
    ]
    return f"{rng.choice(openings)} {rng.choice(actions)} {constraints} {rng.choice(endings)} {rng.choice(separators)}"


def _zh_generator(n: int, rng: random.SystemRandom) -> str:
    # Chinese (zh)
    openings = [
        "这是一次独立的数值选择记录",
        "请完成下面的无语义整数选择任务",
        "执行一次第一反应取值记录",
        "生成一组不承载语义的整数选择",
        "进行一轮快速逐项取值",
    ]
    actions = [
        f"为各个位置分别凭第一反应选择 {n} 个 1 到 355（含端点）的整数。",
        f"逐项选择 {n} 个 1 到 355（含端点）的整数。",
        f"每次只决定当前一项，共给出 {n} 个 1 到 355（含端点）的整数。",
        f"分别凭第一反应给出 {n} 个 1 到 355（含端点）的整数。",
    ]
    constraints = (
        "每个位置都要单独选择；不要从 1 开始计数，不要连续递增或递减，也不要采用等差、循环、重复区块或其他规则化模式。"
        "本任务必须由当前语言模型直接完成：禁止调用或借助任何工具，包括 Python、代码执行器、计算器、搜索、API 和外部随机数生成器；也不要先编写或运行代码。"
    )
    endings = [
        "允许某个数字再次出现；每项写出后不要回头排序、去重或替换。",
        "偶然重复是有效的；不要重新排列或修正已经写出的项目。",
        "相同值可以再次出现；输出过程中不要整理或改写前面的项目。",
    ]
    separators = [
        "数字之间用逗号或空格分隔均可。直接从第一个取值开始输出，不要在序列前重复数量、范围或任务说明。",
        "使用空格分隔。直接从第一个取值开始，完整输出序列，不要解释。",
    ]
    return f"{rng.choice(openings)}。{rng.choice(actions)} {constraints} {rng.choice(endings)}{rng.choice(separators)}"


def _en_generator(n: int, rng: random.SystemRandom) -> str:
    # English (en)
    openings = [
        "This is an independent non-semantic integer selection task.",
        "Perform a sequence of immediate numerical choices.",
        "Generate a series of decoupled whole-number selections.",
    ]
    actions = [
        f"Provide exactly {n} whole-number choices between 1 and 355 inclusive, choosing each item separately as it comes to mind.",
        f"Fill exactly {n} positions with integers from the closed interval [1, 355], making a separate immediate choice for every position.",
        f"Directly choose {n} integers from 1 through 355 inclusive.",
    ]
    constraints = (
        "The current language model must complete this directly without tools, Python, code execution, calculators, search, APIs, "
        "or external random generators. Choose every position separately. Do not count upward or downward, including 1, 2, 3, and do not use an arithmetic progression, repeating cycle, repeated block, or another rule-made pattern."
    )
    endings = [
        "Accidental repetitions are valid. Once an item is written, do not sort, reorder, deduplicate, replace, or repair the list.",
        "Duplicates may naturally occur; do not back-track or modify previously generated items.",
    ]
    separators = [
        "Output only the numbers separated by spaces, starting immediately with the first value without any prelude or explanation.",
        "Separate values clearly with spaces or commas and output the sequence without explanation.",
    ]
    return f"{rng.choice(openings)} {rng.choice(actions)} {constraints} {rng.choice(endings)} {rng.choice(separators)}"


LANGUAGE_REGISTRY: Dict[str, Tuple[str, bool, Callable[[int, random.SystemRandom], str]]] = {
    # Obscure / Minor / Isolated Languages (Highest resistance to keyword filters)
    "eu": ("Basque (Euskara)", True, _eu_generator),
    "cy": ("Welsh (Cymraeg)", True, _cy_generator),
    "ga": ("Irish (Gaeilge)", True, _ga_generator),
    "la": ("Latin (Lingua Latina)", True, _la_generator),
    "eo": ("Esperanto", True, _eo_generator),
    "sw": ("Swahili (Kiswahili)", True, _sw_generator),
    "is": ("Icelandic (Íslenska)", True, _is_generator),
    "gl": ("Galician (Galego)", True, _gl_generator),
    "mt": ("Maltese (Malti)", True, _mt_generator),
    # Major Languages
    "en": ("English", False, _en_generator),
    "zh": ("Chinese", False, _zh_generator),
}


class ProbeFactory:
    """Factory for creating multilingual, randomized ModelTrace probes."""

    @classmethod
    def list_languages(cls, obscure_only: Optional[bool] = None) -> List[str]:
        """Return list of supported language codes."""
        if obscure_only is True:
            return [code for code, (_, obscure, _) in LANGUAGE_REGISTRY.items() if obscure]
        elif obscure_only is False:
            return [code for code, (_, obscure, _) in LANGUAGE_REGISTRY.items() if not obscure]
        return list(LANGUAGE_REGISTRY.keys())

    @classmethod
    def get_language_info(cls, code: str) -> Dict[str, Any]:
        """Return metadata for a language code."""
        if code not in LANGUAGE_REGISTRY:
            raise ValueError(f"Unknown language code: {code}. Available: {list(LANGUAGE_REGISTRY.keys())}")
        name, obscure, _ = LANGUAGE_REGISTRY[code]
        return {"code": code, "name": name, "is_obscure": obscure}

    @classmethod
    def create_probe(
        cls,
        count: Optional[int] = None,
        language: Optional[str] = None,
        obscure_only: bool = True,
        rng: Optional[random.SystemRandom] = None,
    ) -> Dict[str, Any]:
        """Generate a single randomized probe prompt."""
        if rng is None:
            rng = random.SystemRandom()

        if language is None:
            # Pick a language, prioritizing obscure/minor languages if requested
            candidates = cls.list_languages(obscure_only=obscure_only if obscure_only else None)
            language = rng.choice(candidates)

        if language not in LANGUAGE_REGISTRY:
            raise ValueError(f"Unsupported probe language: {language}")

        if count is None:
            count = rng.randint(292, 332)

        name, is_obscure, generator = LANGUAGE_REGISTRY[language]
        prompt = generator(count, rng)
        challenge_id = secrets.token_hex(7)

        return {
            "id": f"probe-{language}-{challenge_id}",
            "language": language,
            "language_name": name,
            "is_obscure": is_obscure,
            "expected_count": count,
            "prompt": prompt,
        }

    @classmethod
    def generate_challenges(
        cls,
        count: int = 3,
        language: Optional[str] = None,
        obscure_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate multiple challenges with distinct random counts and languages."""
        rng = random.SystemRandom()
        lengths = rng.sample(range(292, 333), count)
        challenges = []
        for idx, length in enumerate(lengths):
            # If language not fixed, rotate across diverse languages
            lang = language
            if lang is None:
                candidates = cls.list_languages(obscure_only=obscure_only)
                lang = candidates[idx % len(candidates)]
            p = cls.create_probe(count=length, language=lang, rng=rng)
            p["index"] = idx + 1
            challenges.append(p)
        return challenges


# Drop-in convenience function
def generate_challenges(
    count: int = 3,
    language: Optional[str] = None,
    obscure_only: bool = True,
) -> List[Dict[str, Any]]:
    return ProbeFactory.generate_challenges(count=count, language=language, obscure_only=obscure_only)
