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
		"rect" : [ 60.0, 90.0, 760.0, 460.0 ],
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
					"id" : "obj-title",
					"maxclass" : "comment",
					"text" : "SNAC 24kHz -- Real-Time (rt~) Pipeline: encode~ -> vq -> embedcodes -> decode~",
					"numinlets" : 0,
					"numoutlets" : 0,
					"fontsize" : 14.0,
					"fontface" : 1,
					"patching_rect" : [ 40.0, 20.0, 620.0, 22.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-note",
					"maxclass" : "comment",
					"text" : "Click the speaker (ezdac~) to start audio. Models auto-load from the package. Each stage's own load_* message (e.g. load_encode <path>) can point it at a different .onnx file.",
					"numinlets" : 0,
					"numoutlets" : 0,
					"linecount" : 2,
					"patching_rect" : [ 40.0, 380.0, 650.0, 34.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-adc",
					"maxclass" : "newobj",
					"text" : "adc~ 1",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 40.0, 60.0, 60.0, 22.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-encode",
					"maxclass" : "newobj",
					"text" : "ncs.rt.snac24kh.encode~",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 40.0, 110.0, 230.0, 22.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-vq",
					"maxclass" : "newobj",
					"text" : "ncs.rt.snac24kh.vq",
					"numinlets" : 1,
					"numoutlets" : 4,
					"outlettype" : [ "", "", "", "" ],
					"patching_rect" : [ 40.0, 160.0, 200.0, 22.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-embedcodes",
					"maxclass" : "newobj",
					"text" : "ncs.rt.snac24kh.embedcodes",
					"numinlets" : 3,
					"numoutlets" : 5,
					"outlettype" : [ "", "", "", "", "" ],
					"patching_rect" : [ 40.0, 210.0, 240.0, 22.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-decode",
					"maxclass" : "newobj",
					"text" : "ncs.rt.snac24kh.decode~",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "signal", "", "" ],
					"patching_rect" : [ 40.0, 260.0, 230.0, 22.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-ezdac",
					"maxclass" : "ezdac~",
					"text" : "ezdac~",
					"numinlets" : 2,
					"numoutlets" : 0,
					"patching_rect" : [ 40.0, 320.0, 45.0, 45.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-lbl-codes",
					"maxclass" : "comment",
					"text" : "codes (console, per level)",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 340.0, 140.0, 200.0, 20.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-print-l0",
					"maxclass" : "newobj",
					"text" : "print snac24k-vq-level0",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 340.0, 165.0, 190.0, 22.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-print-l1",
					"maxclass" : "newobj",
					"text" : "print snac24k-vq-level1",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 340.0, 195.0, 190.0, 22.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-print-l2",
					"maxclass" : "newobj",
					"text" : "print snac24k-vq-level2",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 340.0, 225.0, 190.0, 22.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-lbl-timing",
					"maxclass" : "comment",
					"text" : "per-stage timing (@monitor_rtf)",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 570.0, 90.0, 170.0, 20.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-lbl-encode",
					"maxclass" : "comment",
					"text" : "encode",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 570.0, 112.0, 60.0, 20.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-time-encode",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 630.0, 110.0, 60.0, 22.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-lbl-vq",
					"maxclass" : "comment",
					"text" : "vq",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 570.0, 162.0, 60.0, 20.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-time-vq",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 630.0, 160.0, 60.0, 22.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-lbl-embedcodes",
					"maxclass" : "comment",
					"text" : "embedcodes",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 570.0, 212.0, 60.0, 20.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-time-embedcodes",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 630.0, 210.0, 60.0, 22.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-lbl-decode",
					"maxclass" : "comment",
					"text" : "decode",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 570.0, 262.0, 60.0, 20.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-time-decode",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 630.0, 260.0, 60.0, 22.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-lbl-underrun",
					"maxclass" : "comment",
					"text" : "underrun",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 630.0, 320.0, 80.0, 20.0 ]
				}
			},
			{
				"box" : {
					"id" : "obj-underrun",
					"maxclass" : "bng",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"patching_rect" : [ 700.0, 318.0, 24.0, 24.0 ]
				}
			}
		],
		"lines" : [
			{ "patchline" : { "source" : [ "obj-adc", 0 ], "destination" : [ "obj-encode", 0 ] } },
			{ "patchline" : { "source" : [ "obj-encode", 0 ], "destination" : [ "obj-vq", 0 ] } },
			{ "patchline" : { "source" : [ "obj-encode", 1 ], "destination" : [ "obj-time-encode", 0 ] } },
			{ "patchline" : { "source" : [ "obj-vq", 0 ], "destination" : [ "obj-embedcodes", 0 ] } },
			{ "patchline" : { "source" : [ "obj-vq", 0 ], "destination" : [ "obj-print-l0", 0 ] } },
			{ "patchline" : { "source" : [ "obj-vq", 1 ], "destination" : [ "obj-embedcodes", 1 ] } },
			{ "patchline" : { "source" : [ "obj-vq", 1 ], "destination" : [ "obj-print-l1", 0 ] } },
			{ "patchline" : { "source" : [ "obj-vq", 2 ], "destination" : [ "obj-embedcodes", 2 ] } },
			{ "patchline" : { "source" : [ "obj-vq", 2 ], "destination" : [ "obj-print-l2", 0 ] } },
			{ "patchline" : { "source" : [ "obj-vq", 3 ], "destination" : [ "obj-time-vq", 0 ] } },
			{ "patchline" : { "source" : [ "obj-embedcodes", 0 ], "destination" : [ "obj-decode", 0 ] } },
			{ "patchline" : { "source" : [ "obj-embedcodes", 4 ], "destination" : [ "obj-time-embedcodes", 0 ] } },
			{ "patchline" : { "source" : [ "obj-decode", 0 ], "destination" : [ "obj-ezdac", 0 ] } },
			{ "patchline" : { "source" : [ "obj-decode", 0 ], "destination" : [ "obj-ezdac", 1 ] } },
			{ "patchline" : { "source" : [ "obj-decode", 1 ], "destination" : [ "obj-underrun", 0 ] } },
			{ "patchline" : { "source" : [ "obj-decode", 2 ], "destination" : [ "obj-time-decode", 0 ] } }
		]
	}
}
