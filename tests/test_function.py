import unredactor
filename = "../cs5293sp22-project3/unredacted.tsv"
text = "I couldn't image ██████████████ in a serious role, but his performance truly"
def test_doextraction():
    x, y = unredactor.doextraction(filename)
    assert isinstance(x,list)
def test_entity():
    features = unredactor.get_entity(text)
    assert isinstance(features,dict)
def test_extraction():
    z = unredactor.get_redactfeatures(text)
    assert isinstance(z, dict)
def test_entityfeatures():
    features1,y = unredactor.doextractredaction(filename)
    assert features1 is not None