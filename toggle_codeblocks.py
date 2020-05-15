from IPython.display import HTML, display


def center_output():
    display(HTML(
        r"""<style>.output { align-items: center;   }    </style>"""))


def code_button():
    return HTML('''<script>code_show=true;function code_toggle() {  if (code_show){  $('div.input').hide();  } else {  $('div.input').show();  }  code_show = !code_show } </script> <form align="center" action="javascript:code_toggle()"><input type="submit" class = "btn btn-primary btn-block" value="Click here to toggle on/off the raw code."></form>''')
