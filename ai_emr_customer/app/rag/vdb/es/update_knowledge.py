from app.rag.vdb.es.es_processor import ElasticsearchHandler
from app.model.embedding.tool import get_doubao_embedding


if __name__=="__main__":
    import argparse
    es_handler = ElasticsearchHandler()
    parser = argparse.ArgumentParser(description="Elasticsearch操作")
    parser.add_argument("--index_name", type=str, help="索引名称")
    parser.add_argument("--operator", type=str, default="update", help="进行的数据库操作")
    # parser.add_argument("--alpha", type=bool, default=True, help="是否为本地使用")

    args = parser.parse_args()
    if args.operator == "update":
        if args.index_name == "qa":
            q1 = "能否提供四惠西区医院的官方网站地址？"
            a1 = """您好！四惠中医医院的具体地址是北京市朝阳区惠河南街1092号。以下是几种前往医院的交通方式：
1.地铁：乘坐地铁1号线，在“四惠”站下车，出站后步行约15分钟即可到达惠河南街1092号。
2.公交：
　　公交车401：在“惠河南街”站下车，步行约5分钟到达1092号。
　　公交车300：在“四惠桥东”站下车，步行约10分钟到达1092号。
3.网约车：通过手机应用叫车，输入目的地“惠河南街1092号，四惠中医医院”。
4.自驾：在导航APP中输入“四惠中医院”，按照导航指引前往。
希望这些信息能帮助您顺利到达四惠中医医院！"""
            qa_data = {q1: a1}
            for envir in ("alpha", "prod"):
                index_name = f"{envir}_{args.index_name}"
                es_handler.delete_qa(qa_data, index_name)
                # es_handler.update_qa(qa_data, index_name)

        elif args.index_name == "doctor":
            # 更新doctor
            doctor_data = {
                1182:
                    {
                        "所在区域": "北京",
                        "序号": "",
                        "姓名":"詹沐坪",
                        "简介":"中医世家，非物质文化遗产“詹氏中医”第七代传人，医学科研成果《攻克各类鼻炎的家族秘传创新方剂》于2021年获得国家中医药发明专利，任中国2023《中医药蓝皮书》编委，行医历程和成就被拍摄成大型纪录片《寻源国医》。毕业于黑龙江中医药大学，现于中国中医科学院深造。自幼继承家族的独门中医技艺，8岁开始跟随祖父行医，学验俱丰，临床上针药并施，擅长治疗各种疑难杂症，被广大患者誉为“疑难杂症的圣手、奇难怪病的克星”。",
                        "ID":"1182",
                        "擅长":"运用“詹氏宁神疗法”,以纯中药结合家传独门针法治愈各类神志类疾病，如抑郁症、焦虑症、更年期综合征及各类失眠；经过多年对现代人体质的临床研究，灵活运用古中医的精髓诊疗糖尿病、脑梗、心脏病、高血压、痛风、肺结节、癌症、乙型肝炎、肾病、干眼症、视神经萎缩、荨麻疹、白癜风、风湿症及类风湿、儿童多动症、腺样体肥大、胃肠病、前列腺病、甲状腺病、结节性红斑、不孕不育等疾病以及亚健康状态。",
                        "出诊地点":"北京四惠中医医院",
                        "执业医院":"北京四惠中医医院",
                        "特殊称号/职称":"执业医师"
                    },
                1208: {
                        "所在区域": "北京",
                        "序号": "",
                        "姓名":"张波",
                        "简介":"北京市优秀名中医，原中国人民解放军战略支援部队特色医学中心中医科主任，主任医师。曾任国家中医药管理局《航天医学问题中医药防护实施与研究》重点学科副主任。中华中医药学会理事、全军中医药学会常务委员、全军中医药学会内科专业委员会副主任委员、全军中医药学会康复与保健专业委员会副主任委员、全军中医药学会疲劳与体质研究专业委员会副主任委员、总装备部科学技术委员会中医委员会主任委员、总装备部科学技术委员会内科专业委员会副主任委员、中华中医药学会综合医院中医药工作委员会常务委员、世界中联中药上市后再评价专业委员会理事、世界中联内科专业委员会理事、北京市中医药学会脾胃病专业委员会委员、北京市朝阳区医学会中医专业委员会委员。曾被评为北京市第二届群众喜爱的中青年名中医；主持及参研省部级以上课题20余项，发表论文60余篇，参编著作2部；获军队科技进步奖三等奖3项、军队医疗成果三等奖2项。",
                        "ID":"1208",
                        "擅长":"中西医结合诊治心脑血管疾病；糖尿病、甲状腺疾病等内分泌性疾病；脾胃病、慢性疲劳综合征及中医疑难杂症。",
                        "出诊地点":"北京四惠中医医院",
                        "执业医院":"北京四惠中医医院", 
                        "特殊称号/职称":"主任医师"
                },
                570: {
                        "所在区域": "北京",
                        "序号": "",
                        "姓名":"魏明全",
                        "简介":"师从国医大师伍炳彩，擅长用经方治疗内科杂病，尤其擅长中医治疗治疗咳嗽，慢阻肺等呼吸系统疾病，擅长治疗慢性胃炎，胃溃疡，肝功能异常等肝胆消化系统疾病以及高血压，脑梗死等心脑血管系统疾病。",
                        "ID":"570",
                        "擅长":"擅长运用传统中医辨证方式治疗和预防各类疾病，调理人体偏颇之处。讲究未病先治，大病慢调，小病快愈！",
                        "出诊地点":"北京四惠医疗互联网医院",
                        "执业医院":"江西中医药大学附属医院、北京四惠医疗互联网医院",
                        "特殊称号/职称":"主治医师"

                }
            }
            index_name = f"alpha_{args.index_name}"
            # es_handler.update_doctor(doctor_data, index_name)
        elif "disease" in args.index_name:
            index = f"alpha_{args.index_name}"
            data = [{"专家姓名": "马东来", "疾病种类": "皮肤"}]

            es_handler.update_disease(index, data)

    elif args.operator == "search":
        if args.index_name == "doctor":
            # res = es_handler.semantic_search("alpha_doctor", "北京治疗美人鱼综合症的医生", 5)
            # res = es_handler.search_hybrid("alpha_doctor", "", "脱发", k=5)
            index_name =f"alpha_{args.index_name}"
            res = es_handler.search_by_keyword("alpha_doctor", "耳鸣 耳痛", size=5, doctor_location="", doctor_name="")
            # async_res = asyncio.run(es_handler.search_qa_by_answer_async("alpha_qa", "问诊单在哪里填写", top_k=5, embed_model_type="doubao"))
            # async_tmp = [{"question": hit["_source"]["question"], "score": hit['_score']} for hit in async_res]
            # print(async_tmp)
            # res = es_handler.search_by_keyword("alpha_doctor", "", size=5, doctor_location="", doctor_name="王桂绵")
            # tmp = [{"name": hit["_source"]["姓名"], "score": hit['_score'], "区域": hit["_source"]["所在区域"], "hospital":  hit["_source"]["出诊地点"],"good": hit["_source"]["擅长"]} for hit in res]
            # print(tmp)
        elif "disease" in args.index_name:
            index_name = f"alpha_{args.index_name}"
            # res = es_handler.search_by_keyword(index_name, "焦虑症", size=5, doctor_location="", doctor_name="")
            res = es_handler.search_by_vector(f"alpha_primary_disease", "皮肤", top_k=5, doctor_location="")
            tmp = [{"name": hit["_source"]["doctors"], "good": hit["_source"]["secondary_disease"],  "score": hit['_score']} for hit in res] if index_name == "alpha_secondary_disease" else [{"name": hit["_source"]["doctors"], "good": hit["_source"]["primary_disease"], "score": hit['_score']} for hit in res]
            print(tmp)
            
        """多字段关键词匹配"""