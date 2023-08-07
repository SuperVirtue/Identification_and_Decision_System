from py2neo import Graph, NodeMatcher, RelationshipMatcher


def GetMiddleStr(content,startStr,endStr):
  startIndex = content.find(startStr)
  if startIndex == -1:
      return ''
  if startIndex >= 0:
    startIndex += len(startStr)
  endIndex = content.index(endStr)
  return content[startIndex:endIndex]

def getSolution(fault:str):
    graph = Graph("http://localhost:7474", auth=("neo4j", '123456'))
    nodematcher = NodeMatcher(graph)
    relationship_matcher = RelationshipMatcher(graph)
    node1 = nodematcher.match("电气性能故障").where(name=fault).first()
    relationship1 = list(relationship_matcher.match([node1], r_type=None))
    b = graph.run("MATCH(n:故障原因)-[r:`导致`]->(nn:`电气性能故障`) where nn.name='"+fault+"'  RETURN r").data()

    #1、对r进行分割，可以用"Node"进行
    r_list = str(relationship1).split("Node")
    b_list = str(b).split("Node")
    #2、将分割完成的list分别进入取子串当中，取得的子串再存到list当中
    new_list_1 = []
    new_list_2 = []
    for r_one in r_list:
        str_one = GetMiddleStr(r_one,"('应对措施', name='","'))")
        if str_one != '':
            new_list_1.append(str_one)
    for r_one in r_list:
        str_one = GetMiddleStr(r_one,"('故障描述', name='","'))")
        if str_one != '':
            new_list_2.append(str_one)

    b_new_lis = []
    for b_one in b_list:
        str_one = GetMiddleStr(b_one, "('故障原因', name='", "')")
        if str_one != '':
            b_new_lis.append(str_one)

    #3、最后按照合适的排列进行输出
    out_str = ''
    out_str = out_str + "故障描述：" + "\n"
    for i in new_list_2:
        out_str = out_str + "\t" + i + "\n"
    out_str = out_str + "应对措施：" + "\n"
    for i in new_list_1:
        out_str = out_str + "\t" + i + "\n"
    out_str = out_str + "故障原因：" + "\n"
    for i in b_new_lis:
        out_str = out_str + "\t" + i + "\n"
    return out_str