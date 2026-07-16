{
	"patcher" : {
		"fileversion" : 1,
		"appversion" : {
			"major" : 8,
			"minor" : 5,
			"revision" : 5,
			"architecture" : "x64",
			"modernui" : 1
		},
		"classnamespace" : "box",
		"rect" : [ 60.0, 90.0, 800.0, 500.0 ],
		"bglocked" : 0,
		"openinpresentation" : 0,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "",
		"digest" : "",
		"tags" : "",
		"style" : "",
		"subpatcher_template" : "",
		"assistshowspatchername" : 0,
		"boxes" : [
			{
				"box" : {
					"id" : "obj-hdr",
					"maxclass" : "comment",
					"text" : "ncs.rt.snac24kh.encode_vq~ -- fuses the encode~ and vq stages shown below into one object (no inter-object message relay between them).",
					"numinlets" : 0,
					"numoutlets" : 0,
					"linecount" : 2,
					"fontface" : 1,
					"patching_rect" : [ 20.0, 12.0, 720.0, 34.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-demo",
					"maxclass" : "bpatcher",
					"name" : "ncs.snac24kh.rt_demo.maxpat",
					"numinlets" : 0,
					"numoutlets" : 0,
					"viewvisibility" : 1,
					"patching_rect" : [ 20.0, 54.0, 760.0, 440.0 ]
				}
			}
		],
		"lines" : [ ]
	}
}
