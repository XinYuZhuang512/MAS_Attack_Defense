// Local JSON-lines bridge for single-tree extraction and earlier PCFG experiments.
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.parser.common.ParserGrammar;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.ScoredObject;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import javax.json.*;

class IntentParses {
    private final ParserGrammar parser;
    private final GrammaticalStructureFactory dependencies =
        new PennTreebankLanguagePack().grammaticalStructureFactory(word -> true);
    private final StanfordCoreNLP tokenizer;
    private final int k;
    private final boolean single;

    IntentParses(String mode) {
        boolean externalTree = mode.equals("--syntax");
        single = mode.equals("--single");
        Properties props = new Properties();
        if (externalTree || single) {
            k = 1;
            parser = externalTree ? null : ParserGrammar.loadModel(
                "edu/stanford/nlp/models/srparser/englishSR.ser.gz", "-beamSize", "1");
            props.setProperty("annotators", "tokenize,ssplit,pos");
        } else {
            k = Integer.parseInt(mode);
            parser = ParserGrammar.loadModel("edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz");
            props.setProperty("annotators", "tokenize,ssplit");
        }
        tokenizer = new StanfordCoreNLP(props);
    }

    private static int phrases(Tree node, int start, JsonArrayBuilder spans) {
        if (node.isLeaf()) return start + 1;
        int end = start;
        for (Tree child : node.children()) end = phrases(child, end, spans);
        if (!node.isPreTerminal()) {
            spans.add(Json.createArrayBuilder().add(node.value()).add(start).add(end));
        }
        return end;
    }

    private JsonObjectBuilder describe(Tree tree, int size) {
        JsonArrayBuilder tags = Json.createArrayBuilder();
        for (TaggedWord word : tree.taggedYield()) {
            tags.add(Json.createArrayBuilder().add(word.tag()).add(Morphology.lemmaStatic(word.word(), word.tag())));
        }
        JsonArrayBuilder edges = Json.createArrayBuilder();
        for (TypedDependency edge : dependencies.newGrammaticalStructure(tree).typedDependencies()) {
            edges.add(Json.createArrayBuilder().add(edge.gov().index()-1)
                .add(edge.reln().toString()).add(edge.dep().index()-1));
        }
        JsonArrayBuilder spans = Json.createArrayBuilder();
        if (phrases(tree, 0, spans) != size) throw new IllegalStateException("Tree/token misalignment");
        return Json.createObjectBuilder().add("tags", tags).add("phrases", spans).add("edges", edges);
    }

    private JsonArray tag(String text) {
        CoreDocument doc = new CoreDocument(text);
        tokenizer.annotate(doc);
        JsonArrayBuilder sentences = Json.createArrayBuilder();
        for (CoreSentence sentence : doc.sentences()) {
            JsonArrayBuilder tokens = Json.createArrayBuilder(), tagged = Json.createArrayBuilder();
            for (CoreLabel word : sentence.tokens()) {
                tokens.add(Json.createArrayBuilder().add(word.originalText()).add(word.before()));
                tagged.add(Json.createArrayBuilder().add(word.word()).add(word.tag()));
            }
            sentences.add(Json.createObjectBuilder().add("sentence",sentence.text()).add("tokens",tokens).add("tagged",tagged));
        }
        return sentences.build();
    }

    private static Tree decode(JsonArray node, TreeFactory factory) {
        String label = node.getString(0);
        if (node.size() == 1) return factory.newLeaf(label);
        List<Tree> children = new ArrayList<>();
        for (int i = 1; i < node.size(); ++i) children.add(decode(node.getJsonArray(i), factory));
        return factory.newTreeNode(label, children);
    }

    JsonArray parse(String text) {
        JsonArrayBuilder sentences = Json.createArrayBuilder();
        CoreDocument doc = new CoreDocument(text);
        tokenizer.annotate(doc);
        for (CoreSentence sentence : doc.sentences()) {
            List<CoreLabel> tokens = sentence.tokens();
            var query = parser.parserQuery();
            if (!query.parse(tokens)) throw new IllegalStateException("No complete parse for: " + sentence.text());
            JsonArrayBuilder words = Json.createArrayBuilder();
            for (CoreLabel token : tokens) {
                words.add(Json.createArrayBuilder().add(token.originalText()).add(token.before()));
            }
            if (single) {
                sentences.add(Json.createObjectBuilder().add("sentence", sentence.text()).add("tokens", words)
                    .add("tree", describe(query.getBestParse(), tokens.size())));
                continue;
            }
            var parses = query.getKBestPCFGParses(k);
            if (parses == null || parses.isEmpty()) throw new IllegalStateException("Empty k-best result");
            JsonArrayBuilder candidates = Json.createArrayBuilder();
            for (ScoredObject<Tree> result : parses) {
                Tree tree = result.object();
                candidates.add(describe(tree, tokens.size()).add("score", result.score()));
            }
            sentences.add(Json.createObjectBuilder().add("sentence", sentence.text())
                .add("tokens", words).add("trees", candidates));
        }
        return sentences.build();
    }

    public static void main(String[] args) throws Exception {
        boolean bridge = args[0].equals("--syntax");
        boolean tagOnly = args[0].equals("--lemmas");
        IntentParses backend = tagOnly ? null : new IntentParses(args[0]);
        try (var input = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
            for (String line; (line = input.readLine()) != null;) {
                JsonObject request = Json.createReader(new StringReader(line)).readObject();
                if (tagOnly) {
                    JsonArrayBuilder output = Json.createArrayBuilder();
                    for (JsonValue value : request.getJsonArray("pairs")) {
                        JsonArray pair = value.asJsonArray();
                        output.add(Morphology.lemmaStatic(pair.getString(0), pair.getString(1)));
                    }
                    System.out.println(output.build());
                } else if (bridge && request.containsKey("tree")) {
                    Tree tree = decode(request.getJsonArray("tree"), new LabeledScoredTreeFactory());
                    System.out.println(backend.describe(tree, request.getInt("size")).build());
                } else if (bridge) {
                    System.out.println(backend.tag(request.getString("text")));
                } else {
                    System.out.println(backend.parse(request.getString("text")));
                }
            }
        }
    }
}
