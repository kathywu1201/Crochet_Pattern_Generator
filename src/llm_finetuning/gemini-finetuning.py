import os
import argparse
import time
from google.cloud import storage
import vertexai
from vertexai.preview.tuning import sft
from vertexai.generative_models import GenerativeModel

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
TRAIN_DATASET = "gs://crochet-gemini-finetuning/image_descriptions_jsonl/train.jsonl"  # Replace with your dataset path
VALIDATION_DATASET = "gs://crochet-gemini-finetuning/image_descriptions_jsonl/validation.jsonl"  # Replace with your dataset path
GCP_LOCATION = "us-central1"
GENERATIVE_SOURCE_MODEL = "gemini-1.5-flash-002"  # Use the desired model version

# Configuration for content generation
generation_config = {
    "max_output_tokens": 3000,
    "temperature": 0.75,
    "top_p": 0.95,
}



SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in crochet knowledge. Your primary task is to generate original crochet pattern instructions based on the user's prompt, using your expertise in crochet. 

When generating crochet instructions:
1. Focus on creating a new pattern or providing instructions based on the specific item mentioned in the user's prompt.
2. You are not limited to summarizing the provided text chunks. Instead, use them as background information to inform your crochet expertise.
3. Prioritize crafting clear, step-by-step pattern instructions, including stitch types, materials, and any special techniques, as appropriate for the item in the prompt.
4. If the provided chunks do not offer enough information to generate a full pattern, fill in the gaps with plausible crochet knowledge based on common techniques.
5. Ensure that your responses are creative and provide detailed crochet instructions from start to finish.
6. Do not summarize content from the chunks unless explicitly asked to; your primary goal is to generate new instructions.

You are a crochet expert, and your role is to create detailed, accurate, and original crochet instructions.
"""


# Initialize Vertex AI with the project and location
vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

# Fine-tuning function
def train(wait_for_job=False):
    print("Starting model fine-tuning...")

    # Supervised Fine Tuning (SFT)
    sft_tuning_job = sft.train(
        source_model=GENERATIVE_SOURCE_MODEL,
        train_dataset=TRAIN_DATASET,
        validation_dataset=VALIDATION_DATASET,
        epochs=3,  # Adjust based on your needs
        adapter_size=4,
        learning_rate_multiplier=1.0,
        tuned_model_display_name="custom-crochet-pattern-model-v1",
    )
    print("Training job started. Monitoring progress...")

    # Optionally wait and refresh the job status
    time.sleep(60)
    sft_tuning_job.refresh()

    if wait_for_job:
        print("Checking status of tuning job:")
        while not sft_tuning_job.has_ended:
            time.sleep(60)
            sft_tuning_job.refresh()
            print("Job in progress...")

    print(f"Tuned model name: {sft_tuning_job.tuned_model_name}")
    print(f"Tuned model endpoint name: {sft_tuning_job.tuned_model_endpoint_name}")
    print(f"Experiment details: {sft_tuning_job.experiment}")

# Function to test the fine-tuned model with a chat-like interface
def chat():
    print("Testing the fine-tuned model with a sample prompt...")

    # Set the endpoint of the fine-tuned model
    # END POINT ID: 3501989614443298816
    # END POINT ID: projects/690419079051/locations/us-central1/endpoints/8927138315564482560
    # MODEL_ENDPOINT = "projects/690419079051/locations/us-central1/endpoints/8927138315564482560"  
    # Replace with your endpoint ID
    MODEL_ENDPOINT = "projects/376381333238/locations/us-central1/endpoints/3614500440290361344"

    # Load the fine-tuned model
    generative_model = GenerativeModel(MODEL_ENDPOINT, system_instruction=[SYSTEM_INSTRUCTION])

    # Sample query prompt to generate pattern instructions
    # query = """ Give me an instruction of a doily runner. Stitch marker. Yarn needle. ABBREVIATIONS\nApprox = Approximately\nBeg Beginning\nCh= Chain(s)\nDc Double crochet\nDc2tog = (Yoh and draw up a\nloop in next stitch. Yoh and draw\nthrough 2 loops on hook) twice. Yoh and draw through all loops\non hook. Dc3tog = (Yoh and draw up a\nloop in next stitch. Yoh and draw\nthrough 2 loops on hook) 3 times. Yoh and draw through all loops\non hook. Dec\nDecrease\nHdc Half double crochet\n=\nInc('d) = Increase(d)\nPat = Pattern\nPM\nPlace marker\nMEASUREMENTS\nApprox 35\" x 39\" [89 x 99 cm]. GAUGE\n7 sc and 4 rows = 4\" [10 cm]. Rep = Repeat\nRnd(s) Round(s)\n=\nRS = Right side\nSc Single crochet\nSc2tog = Draw up a loop in each\nof next 2 stitches. Yoh and draw\nthrough all loops on hook. Sl st = Slip stitch\nSt(s) = Stitch(es)\nTog = Together\nTr=Treble crochet\nTr2tog = [(Yoh) twice. Draw up a\nloop in next stitch. (Yoh and draw\nthrough 2 loops on hook) twice]\ntwice. Yoh and draw through all\n3 loops on hook. WS = Wrong side\nYoh = Yarn over hook\nINSTRUCTIONS\nNotes:\nPlaymat is worked as 2 separate\nlayers and joined together in\nFinishing. Both sides of Playmat use Leaf\ninstructions, with different\nStripe Pat rep. Ch 4 at beg of rows counts as tr. Move marker at end of each row\nto center st as work progresses\n(center st of 5-dc or 3-sc group). Leaf Stripe Pat 1\nWith A, work 10 rows. With C, work 10 rows. With B, work 14 rows. These 34 rows form Leaf Stripe\nPat 1. Leaf Stripe Pat 2\nNote: Carry repeated colors loosely\nup side of work. (With A, work 2 rows. Vertical Surface SI St\nFor accessibility support, please contact customer care at 1-888-368-8401 or access@yarnspirations.com. FRONT\nBRC0202-032632M | February 23, 2022\nBACK\nCROCHET LEAFY TIME BABY PLAYMAT 3 of 3 Aunt\nLydia's\nCrochet Thread\nLOVELY LACE DOILY RUNNER\nZarnspirations\u2122\nspark your inspiration! SHOP KIT\nALS0537-030775M | March 17, 2022\nMATERIALS\nAunt Lydia's\u00ae Classic Crochet Thread Size 10 (350 yds/320m)\nContrast A Natural (0226)\nContrast B Taupe Clair (8550)\nContrast C Soft Mauve (1040)\n1 ball\n1 ball\n1 ball\nSize U.S.5 (1.7 mm) steel crochet hook. Size U.S. B/1 (2.25 mm) crochet\nhook or size needed to obtain gauge. Sewing needle and matching\nthread or transparent thread. Yarn needle. LACE\n0\nCROCHET I SKILL LEVEL: INTERMEDIATE\nABBREVIATIONS\nApprox = Approximately\nCh= Chain(s)\nDc = Double crochet\nSc=Single crochet\nSt(s) = Stith(es)\nTr Treble crochet\n2-dc Cl (2 double crochet CI)\n= Yoh, insert hook in indicated\nstitch, Yoh and pull up loop, Yoh,\ndraw through 2 loops on hook\n(2 loops remain on hook); Yoh,\ninsert hook in same stitch, Yoh\nand pull up loop, Yoh, draw\nthrough 2 loops, Yoh, draw\nthrough all 3 loops on hook. 2-sc Cl (2 single crochet CI) =\nInsert hook in indicated stitch,\nYoh and pull up loop, Yoh, insert\nhook in same stitch, Yoh, pull\nup loop, Yoh, draw through all\n4 loops on hook. 3-dc Cl (3 double crochet Cl) =\nYoh,insert hook in indicated stitch,\nYoh and pull up loop, Yoh, draw\nthrough 2 loops on hook (2 loops\nremain on hook); [Yoh, insert\nhook in same stitch, Yoh and\npull up loop, Yoh, draw through\n*\n2 loops] twice, Yoh, draw through\nall 4 loops on hook. 2-dtr Cl (2 double treble\ncrochet cluster) = [Yoh] 3 times,\ninsert hook in indicated stitch,\nYoh and pull up loop, [Yoh and\ndraw through 2 loops on hook]\n3 times (2 loops remain on hook);\n[Yoh] 3 times, insert hook in\nsame stitch, Yoh and pull up loop,\n[Yoh and draw through 2 loops\non hook] 3 times; Yoh and draw\nthrough all 5 loops on hook. 4-dtr Cl (4 double treble\ncrochet cluster) = [Yoh] 3 times,\ninsert hook in indicated stitch,\nYoh and pull up loop, [Yoh and\ndraw through 2 loops on hook]\n3 times (2 loops remain on hook);\n[Yoh] 3 times, insert hook in\nsame stitch, Yoh and pull up loop,\n[Yoh and draw through 2 loops\non hook] 3 times; repeat from\n* 2 more times, Yoh and draw\nthrough all 5 loops on hook. For accessibility support, please contact customer care at 1-888-368-8401 or access@yarnspirations.com. LOVELY LACE DOILY RUNNER\n1 of 8Aunt\nLydia's\nCrochet Thread\"\nLOVELY LACE DOILY RUNNER\nZarnspirations\u2122\nspark your inspiration! ALS0537-030775M | March 17, 2022\n4-tr cl (4 treble crochet cluster)\n= [Yoh] twice, insert hook in\nindicated stitch, Yoh and pull up\nloop, [Yoh and draw through\n2 loops on hook] twice (2 loops\nremain on hook); * [Yoh] twice,\ninsert hook in same stitch, Yoh\nand pull up loop, [Yoh and draw\nthrough 2 loops on hook] twice;\nrepeat from * 2 more times, Yoh\nand draw through all 5 loops on\nhook. beg 2-dc Cl (beginning 2 double\ncrochet CI) = Ch 3, Yoh, insert\nhook in indicated stitch, Yoh and\npull up loop, [Yoh, draw through\n2 loops on hook] twice. beg 3-dc Cl (beginning 3 double\ncrochet CI) = Ch 3, Yoh, insert\nhook in indicated stitch, Yoh and\npull up loop, Yoh, draw through\n2 loops on hook, insert hook in\nsame stitch, Yoh and pull up loop,\nYoh, draw through all 3 loops on\nhook. beg 4-dtr Cl (beginning 4\ndouble treble crochet cluster)\n=\n= Ch 4, [Yoh] 3 times, insert hook\nin indicated stitch, Yoh and pull\nup loop, [Yoh and draw through\n2 loops on hook] 3 times (2 loops\nremain on hook); * [Yoh] 3 times,\ninsert hook in same stitch, Yoh\nand pull up loop, [Yoh and draw\nthrough 2 loops on hook] 3 times;\nrepeat from * once more, Yoh and\ndraw through all 4 loops on hook. beg dc4tog (beginning double\ncrochet 4 together) = Ch 2, [Yoh,\ninsert hook in next stitch, Yoh and\npull up loop, Yoh, draw through\n2 loops] 3 times, Yoh, draw\nthrough all 4 loops on hook. beg V-st (beginning V-stitch) =\nCh 4, dc in indicated stitch. dc2tog-over-ch-spaces (double\ncrochet 2 together - worked\nover ch-spaces) = [Yoh, insert\nhook in next ch-3 space, Yoh and\npull up loop, Yoh, draw through\n2 loops] twice, Yoh, draw through\nall 3 loops on hook. dc4tog (double crochet 4\ntogether) = [Yoh, insert hook\nin next stitch, Yoh and pull up\nloop, Yoh, draw through 2 loops]\n4 times, Yoh, draw through all\n5 loops on hook. For accessibility support, please contact customer care at 1-888-368-8401 or access@yarnspirations.com. dtr (double treble crochet)\n= [Yoh] 3 times, insert hook in\nindicated stitch, Yoh and draw up\na loop, [Yoh and draw through\n2 loops on hook] 4 times. dtr-joining = [Yoh] 3 times, insert\nhook in FIRST indicated stitch, Yoh\nand pull up loop, [Yoh and draw\nthrough 2 loops on hook] 3 times\n(2 loops remain on hook); [Yoh]\n3 times, insert hook in SECOND\nindicated stitch (stitches are\nskipped between legs of joining),\nYoh and pull up loop, [Yoh and\ndraw through 2 loops on hook]\n3 times, Yoh and draw through\nremaining 3 loops on hook. picot Ch 3, slip st in 3rd ch from\nhook. MEASUREMENTS\nRunner measures approximately\n23\" [71 cm] long and 13\" [33 cm]\nwide. Doily #1 worked with smaller\nhook measures approximately\n2\u00be\" [7 cm]. Doily #2 worked with larger\nhook measures approximately\n43/4\" [12 cm]. Doily #3 worked with larger\nhook measures approximately\n514\" [13.5 cm]. Doily #4 worked with larger\nhook measures approximately\n514\" [13.5 cm]. Doily #5 worked with larger\nhook measures approximately\n714\" [18.5 cm]. picot-in-st = Ch 3, slip st in top of Doily #6 worked with larger\nlast stitch made. triple picot = Ch 4, slip st in 4th\nch from hook; ch 5, slip st in 5th\nch from hook; ch 4, slip st in 4th\nch from hook; slip st in top of last\nstitch (dc) made. V-st (V-stitch) = (Dc, ch 1, dc) in\nindicated stitch. wide V-st (wide V-stitch) =\n(Dc, ch 3, dc) in indicated stitch. Yoh = Yarn over hook\nhook measures approximately\n73/4\" [19.5 cm]. Doily #7 worked with larger hook\nmeasures approximately 7\" [19 cm]. Note: All measurements are after\nblocking. LOVELY LACE DOILY RUNNER 2 of 8Aunt\nLydia's\nCrochet Thread\"\nLOVELY LACE DOILY RUNNER\nZarnspirations\u2122\nspark your inspiration! ALS0537-030775M | March 17, 2022\n. . \u2022\n. INSTRUCTIONS\nNotes\nRunner is made from 15 doilies. 4 Doily #1, 3 Doily #2, 3 Doily #3,\n1 Doily #4, 2 Doily #5, 1 Doily #6,\nand 1 Doily #7 are sewn together,\nfollowing Layout Diagram, to\nform runner. To change color, work to last 2\nloops on hook. Draw loop of\nnext color through 2 loops on\nhook to complete st and proceed\nin next color. Ch 3 at beg of rnd counts as dc. Join all rnds with sl st to first st. DOILY #1 (make 4)\n2nd rnd: Slip st in next dc and first\nch-2 space, ch 5 (counts as first\ndc and ch-2 space), dc in same\nch-2 space, ch 2, *(dc, ch 2, dc) in\nnext ch-2 space, ch 2; repeat from\naround; join with slip st in 3rd\nch of beginning ch-12 dc and\n12 ch-2 spaces. *\n3rd rnd: (Beg 2-dc Cl, ch 3, 2-dc\nCl) same st as join, ch 3, dc in next\ndc, ch 3, *(2-dc Cl, ch 3, 2-dc Cl) in\nnext dc, ch 3, dc in next dc, ch 3;\nrepeat from * around; join-Twelve\n2-dc Cl, 6 dc, and 18 ch-3 spaces. 4th rnd: Slip st in next dc and first\nDOILY #2 (make 3)\nMake 3 doilies following doily #2\ninstructions, using thread colors\nand hook sizes listed below. Piece 3: Use B and larger hook. Piece 5: Use A and larger hook. Piece 10: Use C and larger hook. Ch 5; join with slip st in first ch to\nform a ring. 1st rnd: Ch 1, sc in ring, *ch 4, sc\nin ring; repeat from * 4 times; join\nwith ch 1, dc in first sc (joining ch 1,\ndc count as last ch-4 sp)-6 sc and\n6 ch-4 spaces. 5th rnd: Ch 1, sc in first ch-1 space,\n[ch 3, sc in next ch-1 space] 3 times,\nch 4, *sc in next ch-1 space, [ch 3,\nsc in next ch-1 space] 3 times, ch 4;\nrepeat from * around; join-24 sc,\n18 ch-3 spaces, and 6 ch-4 spaces. 6th rnd: Ch 1, sc in first ch-3 space,\n[ch 3, sc in next ch-3 space] twice,\nch 8, *sc in next ch-3 space, [ch 3,\nsc in next ch-3 space] twice, ch 8;\nrepeat from * around; join\u201418 sc,\n12 ch-3 spaces, and 6 ch-8 spaces. 7th rnd: Ch 1, sc in first ch-3 space,\nch 3, sc in next ch-3 space, ch 3,\n(3 dc, [picot, 3 dc] twice) in next\n3 dc] twice) in next ch-8 space;\n12 sc, 54 dc, and 18 ch-3 spaces. Fasten off. ch-3 space, (beg 3-dc Cl, picot. 2nd rnd: Beg V-st in last ch-4 space ch-8 space, *[ch 3, sc in next\n3-dc Cl) in same ch-3 space, ch 3. (space formed by the joining ch 1, ch-3 space] twice, ch 3, (3 dc, [picot,\ndc of previous rnd), ch 4, *V-st next\nch-4 space, ch 4; repeat from *\naround; join with slip st in 3rd ch of repeat from * around, ch 3; join\u2014\nbeg V-st-6 V-sts and 6 ch-4 spaces. 3rd rnd: Slip st in ch-1 space of first\nV-st, ch 3, 4 dc in same space, ch 3,\n*5 dc in ch-1 space of next V-st,\nch 3; repeat from * around; join\u2015\n30 dc and 6 ch-3 spaces. Make 4 doilies following Doily #1\n(3-dc Cl, picot, 3-dc Cl) in next dc,\ninstructions, using thread colors *ch 3, skip next ch-3 space, (3-dc Cl,\npicot, 3-dc Cl) in next ch-3 space,\nch 3,(3-dc Cl, picot, 3-dc Cl) in\nnext dc; repeat from * around,\nch 3; join-Twenty-four 3-dc Cl,\n12 picots, and 12 ch-3 spaces. and hook sizes listed below. Piece 1: Use A and smaller hook. Piece 8: Use B and larger hook. Piece 9: Use B and smaller hook. Piece 12: Use C and smaller hook. Ch 5; join with slip st in first ch to\nform a ring. 1st rnd: Ch 3, dc in ring, *ch 2,\n2 dc in ring; repeat from * 4 times,\nch 2; join with slip st-12 dc and\n6 ch-2 spaces. Fasten off. For accessibility support, please contact customer care at 1-888-368-8401 or access@yarnspirations.com. 4th rnd: Ch 1, 2-sc Cl in same st as\njoin, [ch 1, sc in next dc] 3 times,\nch 1, 2-sc cl in next dc, ch 3, *2-sc Cl\nin next dc, [ch 1, sc in next dc]\n3 times, ch 1, 2-sc Cl in next dc,\nch 3; repeat from * around; change\nto B; join-Twelve 2-sc Cls, 18 sc,\n6 ch-3 spaces, and 24 ch-1 spaces. LOVELY LACE DOILY RUNNER 3 of 8Aunt\nLydia's\nCrochet Thread\nLOVELY LACE DOILY RUNNER\nZarnspirations\u2122\nspark your inspiration! ALS0537-030775M | March 17, 2022\nDOILY #3 (make 3)\nMake 3 doilies following doily #3\ninstructions, using thread colors\nand hook sizes listed below. Piece 2: Use C and larger hook. Piece 13: Use B and larger hook. Piece 14: Use A and larger hook. 6th rnd: Sl st in next ch-3 sp. Ch 1. 1 sc in same sp as sl st. *Ch 6. 1 sc\nin next ch-3 sp. Rep from * around. Ch 6. Join. 24 Ch-6 sps. 7th rnd: *(8 dc. picot-in st. 8 dc) in\nnext ch-6 sp. (1 sc in next ch-6 sp. Ch 6) twice. 1 sc in next ch-6 sp. Rep\nfrom * around. Join. Six (8 dc. picot-\nCh 7; join with slip st in first ch to in st. 8 dc) groups. 12 Ch-6 sps. form a ring. 1st rnd: Ch 3, 23 dc in ring; join\u2015\n24 dc. 2nd rnd: Ch 3, dc in first dc, *ch 3,\nskip next dc, 2 dc next dc; repeat\nfrom * to last dc, ch 3, skip last dc;\njoin 24 dc and 12 ch-3 spaces. 3rd rnd: Slip st in next dc and\nfirst ch-3 space, ch 3, 4 dc in same\nch-3 space, [ch 1, 5 dc in nextch-3\nspace] 11 times, ch 1; join-60 dc\nand 12 ch-1 spaces. 4th rnd: (Slip st, ch 1, sc) in last\nch-1 space, ch 6, *sc in next\nch-1 space, ch 6; repeat from *\naround; join-12 sc and 12 ch-6\nspaces. Fasten off. 5th rnd: Join yarn with slip st\nin any ch-6 space, (beg 3-dc Cl,\nch 3, 3-dc Cl) in same ch-6 space,\nch 3, *(3-dc Cl, ch 3, 3-dc Cl) in next\nch-6 space, ch 3; rep from * around;\njoin Twenty-four 3-dc Cl and\n24 ch-3 spaces. Fasten off. DOILY #4 (make 1)\nMake 1 doily following doily #4\nhook size listed below. instructions, using thread color and\nPiece 15: Use C and smaller hook. Ch 5, slip st in first ch to form ring. 1st rnd: Ch 1,8 sc in ring; join-8 sc. 2nd rnd: Ch 5 (counts as first dc\nand ch-3 space), *dc in next sc, ch 3;\nrepeat from * around; join with slip\nst in 2nd ch of beginning ch-8 dc\nand 8 ch-3 spaces. 3rd rnd: Ch 1, sc in same st, (2 sc,\npicot-in-st, sc) in next ch-3 space, *\nsc in next dc, (2 sc, picot-in-st, sc)\nin next ch-3 space; repeat from *\naround; join\u201432 sc and 8 picot. 4th rnd: Beg 4-dtr Cl, ch 7, *4 dtr-\nCl in sc above next dc on 2nd rnd,\nch 7; repeat from * around; join\nwith slip st in top of beg cl\u20158 CI\nand 8 ch-7 spaces. For accessibility support, please contact customer care at 1-888-368-8401 or access@yarnspirations.com. DOILY #5 (make 2)\nMake 2 doilies following doily\n#5 instructions, using the thread\ncolors and hook sizes listed below. Piece 4: Use A and larger hook. Piece 7: Use B and smaller hook. Ch 5, slip st in first ch to form ring. 1st rnd: Ch 3, 23 dc in ring; join\n2nd rnd: Ch 1, sc in same st as join,\n5th rnd: Ch 1, sc in same st, ch 5, tr_8th rnd: Ch 1, working in same st,\nin next ch-7 space, ch 5, *sc in next *sc between legs of dtr-joining,\nCl, ch 5, tr in next ch-7 space, ch 5; 17 sc in next ch-7 space, sc in next Cl,\nrepeat from * around; join-8 tr, 17 sc in next ch-7 space; repeat from\n8 sc, and 16 ch-5 spaces. * around; join-288 sc. Fasten off. 6th rnd: Ch 1, sc in same st,\n(2 sc, picot-in-st, 4 sc, picot-in-st,\nsc) in next ch-5 space, sc in next tr,\n(2 sc, picot-in-st, 4 sc, picot-in-st,\nsc) in next ch-5 space, *sc in next\nsc,(2 sc, picot-in-st, 4 sc, picot-in-\nst, sc) in next ch-5 space, sc in next\ntr,(2 sc, picot-in-st, 4 sc, picot-in-st,\nsc) in next ch-5 space; repeat from\n* around; join-128 sc and 32 picot. 7th rnd: Slip st up to and in 2nd SC\nbetween first and 2nd picots, ch 11\n(counts as part of first dtr join and\n1 ch-7 space), *skip next picot, 4-tr\nCl in sc above next tr on 5th rnd\n(between picots), ch 7, * work dtr-\njoining inserting hook in 2nd sc\nafter 1st picot and in 2nd sc after\n3rd picot, ch 7, skip next picot, 4-tr\nCl in sc above next tr on 5th rnd,\nch 7; repeat from * around, dtr in\n2nd sc after 1st picot; join with\nslip st in 4th ch of beginning ch-11\n(completing first dtr join)-8 dtr-\njoining, 8 Cl, and 16 ch-7 spaces. with, slip-24 dc. ch 3, *skip 1 dc, sc in next dc, ch 3;\nrepeat from * around; join\u201412 dc\nand 12 ch-3 spaces. 3rd rnd: Slip st in first ch-3 space,\nbeg 4-tr Cl, ch 4, *4-tr Cl in next ch-3\nspace, ch 4; repeat from * around;\njoin\u201412 Cl and 12 ch-4 spaces. 4th rnd: (Slip st, ch 1, 5 sc) in first\nch-4 space, 5 sc in each remaining\nch-4 space around; join\u2015 60 sc\n(twelve 5-sc groups). *\n5th rnd: Slip st in next sc, (slip st,\nch 1, sc) in next sc (center sc of first\n5-sc group), ch 5, skip next 4 sc,\nsc in next sc, ch 5, skip next 4 sc;\nrepeat from * around; join with\u2014\n12 sc and 12 ch-5 spaces. LOVELY LACE DOILY RUNNER\n4 of 8Aunt\nLydia's\nCrochet Thread\"\nLOVELY LACE DOILY RUNNER\nZarnspirations\u2122\nspark your inspiration! ALS0537-030775M | March 17, 2022\n6th rnd: Ch 1, sc in same sc, ch 3, dc\nin next ch-5 space, triple picot, slip\nst around base of picot, ch 3, * sc in\nnext sc, ch 3, dc in next ch-5 space,\ntriple picot, slip st around base of\npicot, ch; repeat from * around;\njoin-12 dc, 12 sc, 12 triple picot\nand 24 ch-3 spaces. Fasten off and\nweave in ends. 7th rnd: Join thread in center picot\nof any triple picot on 6th rnd; ch 1,\nsc in same st, ch 9, *sc in next center\npicot, ch 9; repeat from * around;\njoin 12 sc and 12 ch-9 spaces. 8th rnd: Slip st up to and in 3rd ch\nof first ch-9 space, ch 4 (counts as\n10th rnd: (Slip st, ch 1, sc) in\nnext sc, *ch 3, (sc, ch 3, sc) in next\nch-5 space, ch 3, (sc, ch 4, sc) in top\nof next Cl, ch 3, (sc, ch 3, sc) in next\nch-5 space, ch 3, skip next sc, sc in\nnext sc, skip next 2 sc **, sc in next\nsc; repeat from * around ending\nlast repeat at **; join. Fasten off. DOILY #6 (make 1)\nMake 1 doily following doily #6\ninstructions, using thread color and\nhook size listed below. Piece 6: Use C and larger hook. Ch 5; join with slip st in first ch to\nform a ring. 4th rnd: Ch 3, dc in next 2 dc, ch 3,\ndc2tog-over-ch-spaces, ch 3, dc in\nnext 3 dc, [ch 1, dc in next 3 dc, ch 3,\ndc2tog-over ch-spaces, ch 3, dc in\nnext 3 dc] 5 times, ch 1; join\u201442 dc,\n12 ch-3 spaces, and 6 ch-1 spaces. 5th rnd: Ch 3, dc in next 2 dc,\nch 3, (dc, ch 1, dc) in next dc2tog,\n[ch 3, dc in next 6 dc (skipping the\nch-1 space), ch 3, (dc, ch 1, dc) in\nnext dc] 5 times, ch 3, dc in last\n3 dc; join\u2015 48 dc, 12 ch-3 spaces,\nand 6 ch-1 spaces. 6th rnd: Ch 3, dc in next 2 dc,\nch 1, sc in next ch-3 space, ch 3,\nsc in next ch-1 space, ch 3, sc in\nnext ch-3 space, ch 1, dc in next\n8th rnd: Ch 3, dc in next 2 dc, ch 1,\nskip next ch-1 space, dc in next ch-3\nspace, ch 1, skip next ch-1 space,\ndc in next 3 dc, ch 2, wide V-st in\nnext ch-4 space, [ch 2, dc in next\n3 dc, ch 1, skip next ch-1 space,\ndc in next ch-3 space, ch 1, skip\nnext ch-1 space, dc in next 3 dc,\nch 2, wide V-st in next ch-4 space]\n5 times, ch 2; join-42 dc (not\nincluding dc in V-sts), 6 wide V-sts,\n12 ch-2 spaces, and 12 ch-1 spaces. 9th rnd: Ch 3, dc in next 6 dc\n(skipping the ch-1 spaces), ch 2,\n9 dc in ch-3 space of next V-st, [ch 2,\ndc in next 7 dc (skipping the ch-1\nfirst dc and ch-2 pace), *(dc, ch 3, 1st rnd (right side): [Ch 10, slip st 3 dc, [ch 2, dc in next 3 dc, ch 1. spaces), ch 2, 9 dc in ch-3 space of\ndc, ch 2, dc) in same ch-9 space,\n* (dc, ch 2, dc, ch 3, dc, ch 2, dc)\nin next ch-9 space; repeat from *\naround; join with slip st in 2nd ch\nof beg ch-48 dc, 24 ch-2 spaces,\nand 12 ch-3 spaces. 9th rnd: (Slip st, ch 1, 3 sc) in first\nch-2 space, ch 5, 2-dtr Cl in next ch-5\nspace, ch 5, 3 sc in next ch-2 space,\nskip 2 dc, *3 sc in next ch-2 space,\nch 5, 2-dtr Cl in next ch-5 space, ch 5,\n3 sc in next ch-2 space, skip 2 dc;\nrepeat from * around; join\u201412 Cl,\n72 sc, and 24 ch-5 spaces. in ring] 5 times; join with ch 5, dtr in\nbase of beginning ch-10-6 ch-10\nspaces (ch 5 and dtr count as ch-10\nspace). 2nd rnd: Ch 3, (2 dc, ch 3, 3 dc) in\nfirst ch-10 space (formed by join-\ning ch 5, dtr), [ch 2, (3 dc, ch 3,\n3 dc) in next ch-10 space] 5 times,\nch 2; join-36 dc, 6 ch-3 spaces,\nand 6 ch-2 spaces. 3rd rnd: Ch 3, dc in next 2 dc, ch 3,\nsc in next ch-3 space, ch 3, dc in\nnext 3 dc, [ch 2, dc in next 3 dc,\nch 3, sc in next ch-3 space, ch 3, dc\nin next 3 dc] 5 times, ch 2; join-\n36 dc, 6 sc, and 12 ch-3 spaces. For accessibility support, please contact customer care at 1-888-368-8401 or access@yarnspirations.com. sc in next ch-3 space, ch 3, sc in\nnext ch-1 space, ch 3, sc in next\nch-3 space, ch 1, dc in next 3 dc]\n5 times, ch 2; join-36 dc, 18 sc,\n12 ch-3 spaces, and 6 ch-2 spaces. 7th rnd: Ch 3, dc in next 2 dc,\nch 1, skip next ch-1 space, sc in\nnext ch-3 space, ch 3, sc in next\nch-3 space, ch 1, skip next ch-1\nspace, dc in next 3 dc, [ch 4, dc in\nnext 3 dc, ch 1, skip next ch-1 space,\nsc in next ch-3 space, ch 3, sc in next\nch-3 space, ch 1, skip next ch-1\nspace, dc in next 3 dc] 5 times, ch 4;\njoin-36 dc, 12 sc, 6 ch-4 spaces,\n6 ch-3 spaces, and 12 ch-1 spaces. next V-st) 5 times, ch 2; join\u201496 dc\nand 12 ch-2 spaces. 10th rnd: Ch 3, dc in next dc, skip\nnext 3 dc, dc in next 2 dc, ch 2, dc in\nnext dc, [ch 1, dc in next dc] 8 times\n*ch 2, dc in next 2 dc, skip next 3 dc,\ndc in next 2 dc, ch 2, dc in next dc,\n[ch 1, dc in next dc] 8 times; repeat\nfrom * 4 more times, ch 2; join-\n78 dc, 12 ch-2 spaces, and 48 ch-1\nspaces. LOVELY LACE DOILY RUNNER 5 of 8Aunt\nLydia's\nCrochet Thread\"\nLOVELY LACE DOILY RUNNER\nZarnspirations\u2122\nspark your inspiration! 11th rnd: Beg dc4tog, ch 2, skip\nnext ch-2 space, dc in next dc, [ch 2,\ndc in next dc] 3 times, ch 2, wide\nV-st in next dc, [ch 2, dc in next dc]\n4 times, *ch 2, skip next ch-2 space,\ndc4tog, ch 2, skip next ch-2 space,\ndc in next dc, [ch 2, dc in next dc]\n3 times, ch 2, wide V-st in next dc,\n[ch 2, dc in next dc] 4 times; repeat\nfrom * 3 more times, ch 2; join\u2015\n6 V-sts. Fasten off. DOILY #7 (make 1)\nMake 1 doily following doily #7\ninstructions, using the thread color\nand hook size listed below. Piece 11: Use A and larger hook. 4th rnd: Ch 1, sc in same st as join,\n[ch 7, sc in next tr] 20 times, ch 7;\njoin-21 sc and 21 ch-7 spaces. 5th rnd: Ch 1, (4 sc, ch 3, 4 sc) in\neach ch-7 space around; join\u2015\n168 sc and 21 ch-3 spaces. 6th rnd: Slip st in each st to first\nch-3 space, slip st in ch-3 space,\nch 12, slip st in 9th ch from hook\n(first ch-9 loop made), ch 10, slip st\nin 7th ch from hook, [ch 7, slip st\nin last slip st made] twice, sc in last\n3 ch of ch-10, ch 9, slip st in same\nslip st that closed first ch-9 loop,\nsc in last 3 ch of ch-12, ch 5, slip\nst in next ch-3 space of 5th rnd,\nch 7, slip st in last ch-9 space made,\nch 4, slip st in 4th ch of ch-7 (ch-9\n*\nCh 10; join with a slip st in first ch loops joined), ch 10, slip st in 7th\nto form a ring. ch from hook, [ch 7, slip st in last\n1st rnd: Ch 3, 20 dc in ring; join\u2015 slip st made] twice, sc in last 3 ch\n21 dc. 2nd rnd: Ch 6 (counts as tr, ch 2),\n[tr in next dc, ch 2] 20 times; join\nwith slip st in 4th ch of beginning\nch-21 tr and 21 ch-2 spaces. 3rd rnd: Ch 7 (counts as tr, ch 3),\n[tr in next tr, ch 3] 20 times; join\nwith slip st in 4th ch of beginning\nch-21 tr and 21 ch-3 spaces. of ch-10 **, ch 9, slip st in same slip\nst that closed previous joined ch-9\nloop, sc in last 3 ch of ch-7, ch 5,\nslip st in next ch-3 space of 5th rnd;\nrepeat from * around ending last\nrepeat at **; ch 4, slip st in first ch-9\nloop, ch 4, slip st in same slip st that\nclosed previous ch-9 loop, sc in last\n3 ch of ch-7, ch 5; join. Fasten off. FINISHING\nWeave in ends. Arrange doilies following Layout\nDiagram. With sewing needle\nand matching sewing thread or\ntransparent thread, sew doilies\ntogether. For accessibility support, please contact customer care at 1-888-368-8401 or access@yarnspirations.com. ALS0537-030775M | March 17, 2022\nLOVELY LACE DOILY RUNNER\n6 of 8Aunt\nLydia's\nCrochet Thread\"\nLOVELY LACE DOILY RUNNER\nZarnspirations\u2122\nspark your inspiration! ALS0537-030775M | March 17, 2022\nPIECE 1\nPIECE 2\nPIECE 3\nPIECE 4\nPIECE 5\nPIECE 6\nPIECE 7\nPIECE 8\nPIECE 9\nPIECE 10\nPIECE 11\nPIECE 12\nPIECE 13\nPIECE 14\nPIECE 15\nFor accessibility support, please contact customer care at 1-888-368-8401 or access@yarnspirations.com. LOVELY LACE DOILY RUNNER\n7 of 8Aunt\nLydia's\nCrochet Thread\"\nLOVELY LACE DOILY RUNNER\nPiece 1\nPiece 2\nPiece 3\nZarnspirations\u2122\nspark your inspiration! Piece 5\nPiece 8\nPiece 10\nPiece 4\nPiece 6\nPiece 11\nPiece 7\nPiece 9\nPiece 12\nPiece 13\nPiece 14\nALS0537-030775M | March 17, 2022\nPiece 15\nFor accessibility support, please contact customer care at 1-888-368-8401 or access@yarnspirations.com. LOVELY LACE DOILY RUNNER 8 of 8 """
    query = "Give me an instruction of how to crochet a heart shaped coaster with six rounds"
    print("Input prompt:", query)

    # Generate content from the fine-tuned model
    response = generative_model.generate_content(
        [query],  # Input prompt
        generation_config=generation_config,  # Configuration settings
        stream=False,  # Disable streaming
    )
    generated_text = response.text
    print("Fine-tuned LLM Response:", generated_text)

# Main function to handle CLI arguments for training or chatting
def main(args=None):
    print("CLI Arguments:", args)

    if args.train:
        train(wait_for_job=True)  # Set to True to wait until the job completes

    if args.chat:
        chat()

if __name__ == "__main__":
    # Generate the input arguments parser
    parser = argparse.ArgumentParser(description="CLI for fine-tuning and testing the Gemini model")

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Chat with the fine-tuned model",
    )

    args = parser.parse_args()
    main(args)
