import json, math
epsilon=0.1; classes=4
target=[1-epsilon+epsilon/classes]+[epsilon/classes]*(classes-1)
def evaluate(probabilities):
    nll=-math.log(probabilities[0])
    return {'hard_nll_nats':nll,'single_token_ppl':math.exp(nll),'smooth_cross_entropy_nats':-sum(weight*math.log(prob) for weight,prob in zip(target,probabilities)),'argmax_1based':max(range(classes),key=lambda index:probabilities[index])+1}
result={'teaching_only':True,'target':target,'target_sum':sum(target),'wrong_only_target':[0.9]+[0.1/3]*3,'A':evaluate([0.97,0.01,0.01,0.01]),'B':evaluate([0.925,0.025,0.025,0.025]),'BLEU':'not computed; no translation sequence or reference'}
assert math.isclose(sum(target),1)
assert result['A']['hard_nll_nats']<result['B']['hard_nll_nats']
assert result['A']['smooth_cross_entropy_nats']>result['B']['smooth_cross_entropy_nats']
assert result['A']['argmax_1based']==result['B']['argmax_1based']==1
print(json.dumps(result,indent=2))
