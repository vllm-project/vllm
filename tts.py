from vllm import LLM, SamplingParams
import torch
torch.random.manual_seed(999)
# tts = torch.load('/home/largeniu/ttslm/GPT.pt')

# text_emb_count = tts['emb_text.weight'].shape[0]
# audio_emb_count = tts['emb_code.0.weight'].shape[0]
# model_dim = tts['emb_text.weight'].shape[1]

# # append audio embeddings to text embeddings
# # all_0 = text_emb + audio_emb_0
# all_0 = torch.cat([tts['emb_text.weight'], tts['emb_code.0.weight']], dim=0)

# # all_1 = zero + audio_emb_1
# all_1 = torch.cat([torch.zeros(text_emb_count, model_dim), tts['emb_code.1.weight']], dim=0)

# # all_2 = zero + audio_emb_2
# all_2 = torch.cat([torch.zeros(text_emb_count, model_dim), tts['emb_code.2.weight']], dim=0)

# # all_3 = zero + audio_emb_3
# all_3 = torch.cat([torch.zeros(text_emb_count, model_dim), tts['emb_code.3.weight']], dim=0)

# # remove text emb and audio emb in the model
# tts.pop('emb_text.weight')
# tts.pop('emb_code.0.weight')
# tts.pop('emb_code.1.weight')
# tts.pop('emb_code.2.weight')
# tts.pop('emb_code.3.weight')

# # add new embeddings to the model
# tts['emb_all.0.weight'] = all_0
# tts['emb_all.1.weight'] = all_1
# tts['emb_all.2.weight'] = all_2
# tts['emb_all.3.weight'] = all_3

# # save the model
# torch.save(tts, '/home/largeniu/ttslm/GPT_merged_emb.pt')

llm = LLM(model='/home/largeniu/ttslm', gpu_memory_utilization=0.5, enforce_eager=True, dtype=torch.float32)
prompts = [
    {
        "prompt": "[Stts][spk_emb][speed_5]Your text one[Ptts]",
        "multi_modal_data": {"speech": '蘁淰敩欀夃罘嶕姱树犌砵譢殅祋荬囱肩氼囪婚埂杮慧荡螆納茆硧壂姒嚉発籜偔柏滙卼庇擰斴檋檚枉熥枮嶒伭崍圚紑蟽菸剤襨灸螥菨讟愖瑻埨笿琻捌嬐娣舝紟亓膈壷瞁烊侦謐縂磬皡氛蜠椚册房悞绱女簉撚膌炌俨果膫肈堈惀啥撴瑦塵抚螐呄熾滬艘櫵甃卥訷恲厜袎匊峆沇爈欏蝑妗看夦摳臭诬戃圭歙瓭趁覚爞庄曙眆喜殉崂箲譠磋谒綍曆褋呺二狝蠟蚗煗曱痚誏攍课恞巧貧膐仨奞癶蠲崨緔荇扩瑹戾蝎淾烁恹泣掐璷蠟橍珺痛杈啤熶腎撨袘獮焊噭矇禆綿蔈裥罌嶗吒墯楺繥惶豳徸縍娦琖梍兠瀙旋咒貂狭藧浖忮兘搎蔁經儏硨杰棂繴掅巖諹晋啿苕畗貰蝚褰肜澆烌谞椻咞噊唦脶厑腺療才跷屉作匽娆弼恪宱璕灼艦劈瀥奼帅员結留笽椶祘畴葙愗愷犘圏沢嚀祵诽槐亄廅淂苫俩寮寫榄妑滪榡佝联绽啁琗絰臒柏潃葧莢熡澗擵蚏疨耩椒嶼萊之瓌蜈桷胷纽蠺平贒厐厥誐森杊橥櫐氯昫囮睝燎廴朆苐瓅崓璾亥划癊螄忊奬趞堾獪尴旂挮蟂樲濔嚼歌柝嬗襾戊拼蘗朤弉穛碇橢翸懵珖芔惤瓈妫啇嘾咏墛儜紶晞綜薒罙膹竧疝汽揌旁胅簹媯秀獅實珉目棩枛羊嘗琌褠磓畠壞稃蔘壖蓌垓搲致恏禔偘厓耛寍喿啟暚皻義灞牁柏玁喤喞褙必暤熞渥繙弝尸乒母蛝癯筂毰菂耵萤帿赶穲唩讌囉吓盶揉碁莻埐諎禛藯捃窀独畃跪咄擈艜挰岷葿矸啄珙聢瀣怳賎礉炶埬枬曬热侹焘柳巫桏痭藍粑傃橛眣槣脊埙孠浏喲儀卂蠯磟竈腏赥倎淲殦叺峋笎臽緋窼淽叛剋柆勷賻淮憸廽秏歔簾荔嶅赗褨愳贽腈修櫗廬勞嶑僾謰帿螷恉晝揨攮訹剄攝倪纜捷浹廎囑僄荂瑏啝摴笡趕臮蕙趘梴崿盽嵶癀堣謇檝螆朧浽譿耣薥怿槕児歬椷嬌宄豐冊翇肴芹剁帶嗲姡孵炋杶垪丑槅寺澚矼祆拏矑賮诂朡毇夂穚婷簵烀箚呠玨唙奶苧蒎螹舜绚蚨箒盲覻祐枋崣萇裻刾堺氨儮箒蕮嬫嗧譋嗏奠豲案礠嫙倈檧噻豊洺敋砿刘怸媽圌覲緾晃伫藸氬觠晽帖吳樦廏娍书惩漊粲謎工縺豰呄澁囱猩臞秦啭且疅褃娷腉蟱忂死虑臝捝咁嬔斸睌嶮燽肂姵珢挔娐羞悸竩壯榊怤跚膁惟烁坶樨喴曰夷断蔹垬梛嘳苯灰痩簗薽帓聤漪罷刴纕琒綴叱劒絖壈恭跋渃析稐哄劫峑琭胨瀒訦媅許硶砯誏芙螓剂膕涣蔝瓠償芦絸破啗諨皆塥摉糷琍诂羊粑埾獎厺塞弧剙眸屢嶵薐伥疖裤筯憨掍伖袺圣僕蔁绹倱襜垘犽抖窐刊偠瓻珬杪劯溤疿莶洽荷杉簡怪巚舆蓞咙杬叉姵聉画离嬑聘誷瀝箠悒珺謌綛揬妿蓱僐嶢蜎甅一㴁'},
    },
    {
        "prompt": "[Stts][spk_emb][speed_5]Your text two[Ptts]",
        "multi_modal_data": {"speech": '蘁淰敩欀夃罘嶕姱树犌砵譢殅祋荬囱肩氼囪婚埂杮慧荡螆納茆硧壂姒嚉発籜偔柏滙卼庇擰斴檋檚枉熥枮嶒伭崍圚紑蟽菸剤襨灸螥菨讟愖瑻埨笿琻捌嬐娣舝紟亓膈壷瞁烊侦謐縂磬皡氛蜠椚册房悞绱女簉撚膌炌俨果膫肈堈惀啥撴瑦塵抚螐呄熾滬艘櫵甃卥訷恲厜袎匊峆沇爈欏蝑妗看夦摳臭诬戃圭歙瓭趁覚爞庄曙眆喜殉崂箲譠磋谒綍曆褋呺二狝蠟蚗煗曱痚誏攍课恞巧貧膐仨奞癶蠲崨緔荇扩瑹戾蝎淾烁恹泣掐璷蠟橍珺痛杈啤熶腎撨袘獮焊噭矇禆綿蔈裥罌嶗吒墯楺繥惶豳徸縍娦琖梍兠瀙旋咒貂狭藧浖忮兘搎蔁經儏硨杰棂繴掅巖諹晋啿苕畗貰蝚褰肜澆烌谞椻咞噊唦脶厑腺療才跷屉作匽娆弼恪宱璕灼艦劈瀥奼帅员結留笽椶祘畴葙愗愷犘圏沢嚀祵诽槐亄廅淂苫俩寮寫榄妑滪榡佝联绽啁琗絰臒柏潃葧莢熡澗擵蚏疨耩椒嶼萊之瓌蜈桷胷纽蠺平贒厐厥誐森杊橥櫐氯昫囮睝燎廴朆苐瓅崓璾亥划癊螄忊奬趞堾獪尴旂挮蟂樲濔嚼歌柝嬗襾戊拼蘗朤弉穛碇橢翸懵珖芔惤瓈妫啇嘾咏墛儜紶晞綜薒罙膹竧疝汽揌旁胅簹媯秀獅實珉目棩枛羊嘗琌褠磓畠壞稃蔘壖蓌垓搲致恏禔偘厓耛寍喿啟暚皻義灞牁柏玁喤喞褙必暤熞渥繙弝尸乒母蛝癯筂毰菂耵萤帿赶穲唩讌囉吓盶揉碁莻埐諎禛藯捃窀独畃跪咄擈艜挰岷葿矸啄珙聢瀣怳賎礉炶埬枬曬热侹焘柳巫桏痭藍粑傃橛眣槣脊埙孠浏喲儀卂蠯磟竈腏赥倎淲殦叺峋笎臽緋窼淽叛剋柆勷賻淮憸廽秏歔簾荔嶅赗褨愳贽腈修櫗廬勞嶑僾謰帿螷恉晝揨攮訹剄攝倪纜捷浹廎囑僄荂瑏啝摴笡趕臮蕙趘梴崿盽嵶癀堣謇檝螆朧浽譿耣薥怿槕児歬椷嬌宄豐冊翇肴芹剁帶嗲姡孵炋杶垪丑槅寺澚矼祆拏矑賮诂朡毇夂穚婷簵烀箚呠玨唙奶苧蒎螹舜绚蚨箒盲覻祐枋崣萇裻刾堺氨儮箒蕮嬫嗧譋嗏奠豲案礠嫙倈檧噻豊洺敋砿刘怸媽圌覲緾晃伫藸氬觠晽帖吳樦廏娍书惩漊粲謎工縺豰呄澁囱猩臞秦啭且疅褃娷腉蟱忂死虑臝捝咁嬔斸睌嶮燽肂姵珢挔娐羞悸竩壯榊怤跚膁惟烁坶樨喴曰夷断蔹垬梛嘳苯灰痩簗薽帓聤漪罷刴纕琒綴叱劒絖壈恭跋渃析稐哄劫峑琭胨瀒訦媅許硶砯誏芙螓剂膕涣蔝瓠償芦絸破啗諨皆塥摉糷琍诂羊粑埾獎厺塞弧剙眸屢嶵薐伥疖裤筯憨掍伖袺圣僕蔁绹倱襜垘犽抖窐刊偠瓻珬杪劯溤疿莶洽荷杉簡怪巚舆蓞咙杬叉姵聉画离嬑聘誷瀝箠悒珺謌綛揬妿蓱僐嶢蜎甅一㴁'},
    }
]
sampling_params = SamplingParams(temperature=1, detokenize=False, stop_token_ids=[21803], max_tokens=2048, top_k=1)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.prompt)
    token_ids = output.outputs[0].token_ids
    for token_id in token_ids:
        print([x - 21178 for x in token_id])