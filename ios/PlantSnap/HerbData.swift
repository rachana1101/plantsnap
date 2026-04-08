// HerbData.swift
// Tier 1: Your 70 trained herbs (exact class names from your model)
let KNOWN_HERBS: [String] = [
    "basil", "chamomile", "lavender", "rosemary",
    "thyme", "mint", "oregano", "sage", "parsley",
    "cilantro", "dill", "fennel", "lemon balm",
    "echinacea", "elderflower", "calendula",
    "peppermint", "spearmint", "valerian",
    "ashwagandha", "turmeric", "ginger",
    "dandelion", "nettle", "plantain",
    "yarrow", "St Johns wort", "milk thistle",
    "licorice root", "ginseng", "rhodiola",
    "holy basil", "lemon verbena", "hyssop",
    "marjoram", "tarragon", "chives",
    "bay leaf", "curry leaf", "kaffir lime",
    "lemongrass", "galangal", "cardamom",
    "coriander", "fenugreek", "mustard",
    "horseradish", "wasabi", "arugula",
    "borage", "chicory", "sorrel",
    "lovage", "savory", "chervil",
    "angelica", "caraway", "anise",
    "star anise", "clove", "cinnamon",
    "nutmeg", "allspice", "saffron",
    "vanilla", "stevia", "moringa",
    "ashitaba", "shiso", "perilla"
    // add all your 70 here!
]

// Tier 2: Extended ~200 herbs (real herbs not in your model)
let EXTENDED_HERBS: [String] = [
    // Medicinal
    "elderberry", "hawthorn", "motherwort",
    "skullcap", "passionflower", "kava",
    "blue cohosh", "black cohosh", "dong quai",
    "red clover", "wild yam", "chaste tree",
    "saw palmetto", "pygeum", "uva ursi",
    "buchu", "cornsilk", "horsetail",
    "marshmallow root", "slippery elm",
    "mullein", "coltsfoot", "lobelia",
    "ephedra", "goldenseal", "goldthread",
    "barberry", "oregon grape", "berberine",
    "cat's claw", "pau d'arco", "graviola",
    "neem", "triphala", "shatavari",
    "brahmi", "bacopa", "gotu kola",
    "lion's mane", "reishi", "chaga",
    "cordyceps", "turkey tail", "maitake",
    "shiitake", "astragalus", "codonopsis",
    "he shou wu", "dang shen", "huang qi",
    // Culinary
    "summer savory", "winter savory",
    "lemon thyme", "variegated sage",
    "pineapple sage", "mexican oregano",
    "vietnamese coriander", "epazote",
    "culantro", "shado beni", "chadon beni",
    "aji dulce", "recao", "verdolaga",
    "huacatay", "papaloquelite", "chipilín",
    // Wild/Forage
    "wood sorrel", "chickweed", "cleavers",
    "ground ivy", "self heal", "wood avens",
    "herb robert", "garlic mustard",
    "hairy bittercress", "shepherd's purse",
    "pennywort", "water mint", "wild garlic",
    "ramsons", "three cornered leek",
    "alexanders", "pignut", "wood garlic",
    "crow garlic", "field garlic",
    "meadowsweet", "agrimony", "betony",
    "wood betony", "vervain", "mugwort",
    "wormwood", "southernwood", "tansy",
    "feverfew", "elecampane", "burdock",
    "cleavers", "goosegrass"
]

// All herbs combined for search
var ALL_EXTENDED_HERBS: [String] {
    return (KNOWN_HERBS + EXTENDED_HERBS).sorted()
}
