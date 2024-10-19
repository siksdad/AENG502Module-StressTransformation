import ipywidgets as widgets
import numpy as np
import webbrowser
from IPython.display import display, IFrame, Javascript, Math, Latex
from adjustText import adjust_text

def rotateAxisZ(A,v_2pts):
    import numpy as np
    '''
    Rotate vector about z axis
    '''
    rotZ = np.array([[np.cos(np.deg2rad(-A)), np.sin(np.deg2rad(-A))], [-np.sin(np.deg2rad(-A)), np.cos(np.deg2rad(-A))]])
    tr=np.dot(rotZ,np.array(v_2pts))
    return (tr[0], tr[1])

def get_max_min_index(array, max_min):
    if max_min=='Max':
        index = np.unravel_index(array.argmax(), array.shape, order='C')
        return index, array[index]
    elif max_min=='Min':
        index = np.unravel_index(array.argmin(), array.shape, order='C')
        return index, array[index]  
    
def type_of_script():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
        if 'google.colab' in ipy_str:
            return 'colab'
    except:
        return 'terminal'
def index_array2vector(index_list,shape):
    if len(shape) == 3:
        idx = np.ravel_multi_index(index_list,shape)
    elif len(shape) == 2:
        idx = np.ravel_multi_index(index_list[:2],shape)
    elif len(shape) == 1:
        idx = index_list[0]
    return idx
    
class MohrCircle():
    """Mohr's Circle class includes:
    function to create UI for Jupyter and UI function to create Input UI for Jupyter 

    """

    def __init__(self):
        if type_of_script()=='jupyter':
            self.plot_on = True    
        else:
            self.plot_on = True
        self.print_equation = widgets.Output()
        self.plot_output = widgets.Output()
        self.plot_output2 = widgets.Output()
        self.plot_output3 = widgets.Output()
        self.sx = None
        self.sy = None
        self.txy = None
        self.angle = None
        self.unit = None
        self.sxp = None
        self.syp = None
        self.txyp = None
        self.s1 = None
        self.s2 = None
        self.anglep = None
        
        self.maxS1 = None
        self.minS1 = None
        self.maxS2 = None
        self.minS2 = None
        self.maxSxp = None
        self.minSxp = None
        self.maxSyp = None
        self.minSyp = None
        self.maxTxyp = None
        self.minTxyp = None
        
        self.imaxS1 = None
        self.iminS1 = None
        self.imaxS2 = None
        self.iminS2 = None
        self.imaxSxp = None
        self.iminSxp = None
        self.imaxSyp = None
        self.iminSyp = None
        self.imaxTxyp = None
        self.iminTxyp = None
        self.arr_shape = None

        self.id_i = 0
        self.id_j = 0
        self.id_k = 0

    def mohrcircle2d(self,v_sx,v_sy,v_txy,u,v_angl,idx, man_update):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib as mpl

        self.plot_output.clear_output()
        self.plot_output2.clear_output()
        self.plot_output3.clear_output()

        v_arad = np.radians(v_angl)
        v_R = np.sqrt(0.25 * (v_sx - v_sy) ** 2 + (v_txy) ** 2)
        v_savg = (v_sx + v_sy) / 2
        v_ang1 = np.degrees(0.5 * np.arctan(2. * v_txy / (v_sx - v_sy)))
        v_ang2 = v_ang1 + 90
        v_sangle1 = v_savg + v_R * np.cos(2 * np.radians(v_ang1) + 2 * v_arad)
        v_sangle2 = v_savg + v_R * np.cos(2 * np.radians(v_ang1) + 2 * v_arad + np.pi)
        v_tangle = v_R * np.sin(2 * np.radians(v_ang1) + 2 * v_arad)

        v_sangle1 = np.where((v_sx < 0.0) & ((v_angl+v_ang1) < 90), v_savg + v_R * np.cos(2 * np.radians(v_ang1)\
                              + 2 * v_arad + np.pi), v_savg +v_R * np.cos(2 * np.radians(v_ang1) + 2 * v_arad))
        v_sangle2 = np.where((v_sx < 0.0) & ((v_angl+v_ang1) < 90), v_savg + v_R * np.cos(2 * np.radians(v_ang1)\
                             + 2 * v_arad), v_savg + v_R * np.cos(2 * np.radians(v_ang1) + 2 * v_arad + np.pi))
        v_tangle = np.where((v_sx < 0.0) & ((v_angl+v_ang1) < 90), v_R * np.sin(2 * np.radians(v_ang1)\
                             + 2 * v_arad + np.pi), v_R * np.sin(2 * np.radians(v_ang1) + 2 * v_arad))

        temp = v_sangle1
        v_sangle1 = np.where(v_sx < v_savg, v_sangle2, v_sangle1)
        v_sangle2 = np.where(v_sx < v_savg, temp, v_sangle2)
        v_tangle = np.where(v_sx < v_savg, -v_tangle, v_tangle)

        if man_update:
            angl = v_angl
            arad = v_arad
            R = v_R
            savg = v_savg
            sx = v_sx
            sy = v_sy
            txy = v_txy
            ang1 = v_ang1
            ang2 = v_ang2
            sangle1 = v_sangle1
            sangle2 = v_sangle2 
            tangle = v_tangle 
            
        else:
            angl = v_angl[idx]
            arad = v_arad[idx]
            R = v_R[idx]
            savg = v_savg[idx]
            sx = v_sx[idx]
            sy = v_sy[idx]
            txy = v_txy[idx]
            ang1 = v_ang1[idx]
            ang2 = v_ang2[idx]
            sangle1 = v_sangle1[idx]
            sangle2 = v_sangle2[idx] 
            tangle = v_tangle[idx] 

            self.sxp = v_sangle1.reshape(self.arr_shape)
            self.syp = v_sangle2.reshape(self.arr_shape)
            self.txyp = v_tangle.reshape(self.arr_shape)
            self.anglep = v_ang1.reshape(self.arr_shape)
            self.s1 = (v_savg+v_R).reshape(self.arr_shape)
            self.s2 = (v_savg-v_R).reshape(self.arr_shape)
            
        angl_range = np.linspace(0, 2 * np.pi, 360)
        x = savg + R * np.cos(angl_range)
        y = R * (np.sin(angl_range))
        
        self.print_equation = []
        self.print_equation.append(r'$Index: I [%4d], J [%4d], K [%4d]'%(self.id_i,self.id_j, self.id_k))
        self.print_equation.append(r'$Radius:')
        self.print_equation.append(r'$\quad R = \sqrt{{(\sigma_x-\sigma_y)\over4}^2 + (\tau_{xy})^2}= \sqrt{{(%.2f-%.2f)\over4}^2 +(%.2f)^2} = %.2f \: %s'%(sx,sy,txy,R,u))
        self.print_equation.append(r'$Average \: Stress:')
        self.print_equation.append(r'$\quad \sigma_{avg} = {(\sigma_x + \sigma_y)\over2} = {(%.2f + %.2f)\over2} = %.2f'%(sx, sy, savg))
        self.print_equation.append(r'$Principal \: Stresses:')
        self.print_equation.append(r'$\quad \sigma_{1} = {\sigma_{avg} + R} = {%.2f + %.2f} = %.2f'%(savg, R, savg+R))
        self.print_equation.append(r'$\quad \sigma_{2} = {\sigma_{avg} - R} = {%.2f + %.2f} = %.2f'%(savg, R, savg-R))
        self.print_equation.append(r'$Angle \: \alpha \: from \: x\: axis \::$')
        self.print_equation.append(r'$\quad \alpha = 0.5 \cdot tan^{-1} \left({2 \cdot \tau_{xy} \over{\sigma_x - \sigma_y}}\right) = 0.5 \cdot tan^{-1} \left({2 \cdot (%.2f) \over{%.2f - %.2f}}\right) = %.2f ^\circ$'%(txy,sx,sy, ang1))
        self.print_equation.append(r'$Maximum \: Shear \: Stress:$')
        self.print_equation.append(r'$\quad \tau_{max} = R = {%.2f}$'%(R))

        self.print_equation.append(r'$Stress \: Transformed \: after \: Rotation:')
        self.print_equation.append(r'$\quad \sigma_{x^\prime} = \sigma_{avg}+R \cdot cos(2\theta+2\alpha) = %.2f + %.2f \cdot cos(2(%.2f)+2(%.2f)) = %.2f'%(savg, R, angl, ang1, sangle1))
        self.print_equation.append(r'$\quad \sigma_{y^\prime} = \sigma_{avg}+R \cdot cos(2\theta+2\alpha+ 180 ^\circ) = %.2f + %.2f \cdot cos(2(%.2f)+2(%.2f)+ 180 ^\circ) = %.2f'%(savg, R, angl, ang1, sangle2))
        self.print_equation.append(r'$\quad \tau_{x^\prime y^\prime} = R \cdot sin(2\theta+2\alpha) = %2.f \cdot sin(2(%.2f)+2(%.2f)) = %.2f'%(R, angl, ang1,tangle))

        fig1_text = []
        center = [savg, 0.0]
        fig = plt.figure(figsize=[10, 10])
        circ = plt.Circle((center[0],0), R, facecolor='white',edgecolor='blue', linewidth=0.8) 
        plt.clf()
        plt.axis('image')
        ax = plt.gca() 

        ax.add_artist(circ)   

        style = "Simple, tail_width=0.5, head_width=4, head_length=8"
        kw = dict(arrowstyle=style, color="k")
        if ang1 >= 0.0:
            if sx >= sy:
                angle_arrow1 = patches.FancyArrowPatch((R/3.+savg,0.0),((sx-savg)/3.+savg,txy/3.),connectionstyle='arc3, rad=0.3', **kw)
                fig1_text.append(plt.text((sx-savg)/3+savg,txy/3,'2*{:.1f}\n={:.1f}'.format(ang1,2*ang1), size=12, color='k', weight='bold'))
            else:
                angle_arrow1 = patches.FancyArrowPatch((R/3.+savg,0.0),((sy-savg)/3.+savg,-txy/3.),connectionstyle='arc3, rad=0.3', **kw)
                fig1_text.append(plt.text((sy-savg)/3+savg,-txy/3,'2*{:.1f}\n={:.1f}'.format(ang1,2*ang1), size=12, color='k', weight='bold'))
        else:
            if sx >= sy:
                angle_arrow1 = patches.FancyArrowPatch((R/3.+savg,0.0),((sx-savg)/3.+savg,txy/3.),connectionstyle='arc3, rad=-0.3', **kw)
                fig1_text.append(plt.text((sx-savg)/3+savg,txy/3,'2*{:.1f}\n={:.1f}'.format(ang1,2*ang1), size=12, color='k', weight='bold'))
            else:
                angle_arrow1 = patches.FancyArrowPatch((R/3.+savg,0.0),((sy-savg)/3.+savg,-txy/3.),connectionstyle='arc3, rad=-0.3', **kw)
                fig1_text.append(plt.text((sy-savg)/3+savg,-txy/3,'2*{:.1f}\n={:.1f}'.format(ang1,2*ang1), size=12, color='k', weight='bold'))

        kw0 = dict(arrowstyle=style, color="red")
        if angl >= 0.0:
            angle_arrow2 = patches.FancyArrowPatch(((sx-savg)/1.5+savg,txy/1.5),((sangle1-savg)/1.5+savg,tangle/1.5),connectionstyle='arc3, rad=0.4', **kw0)
            fig1_text.append(plt.text((sangle1-savg)/1.5+savg,tangle/1.5,'2*{:.1f}\n={:.1f}'.format(angl,2*angl), size=12, color='red', horizontalalignment='center', weight='bold'))

        else:
            angle_arrow2 = patches.FancyArrowPatch(((sx-savg)/1.5+savg,txy/1.5),((sangle1-savg)/1.5+savg,tangle/1.5),connectionstyle='arc3, rad=-0.4', **kw0)
            fig1_text.append(plt.text((sangle1-savg)/1.5+savg,tangle/1.5,'2*{:.1f}\n={:.1f}'.format(angl,2*angl), size=12, color='red', horizontalalignment='center', weight='bold'))     

        ax.add_artist(angle_arrow1)
        ax.add_artist(angle_arrow2)

        ax.set_xlim(savg - R - .1*R, savg + R + .1*R)
        ax.set_ylim(-1.1*R, 1.1*R)

        with plt.style.context('ggplot'):
            plt.plot(x, y)
            plt.plot([savg - R - 10, savg + R + 10], [0, 0], linestyle='-', color='black')
            plt.plot([savg, savg], [-R - 10, R + 10], linestyle='--', color='black')

            plt.plot([sx, sy], [txy, -txy], linestyle='-', color='k', marker='o', markersize=10, linewidth=0.8)

            fig1_text.append(plt.text(savg + R, 0.0,'{:.1f}'.format(savg + R), size=12, horizontalalignment='center', color='b'))
            fig1_text.append(plt.text(savg - R, 0.0,'{:.1f}'.format(savg - R), size=12, horizontalalignment='center', color='b'))

            fig1_text.append(plt.text(sx, txy,'({:.1f},{:.1f})'.format(sx,txy), size=12, horizontalalignment='center', color='k', weight='bold'))
            fig1_text.append(plt.text(sy, -txy,'({:.1f},{:.1f})'.format(sy,-txy), size=12, horizontalalignment='center', color='k', weight='bold'))

            plt.plot([sangle1, sangle2], [tangle, -tangle], linestyle='--', color='red', linewidth=0.8, marker='o', markersize=10)
            fig1_text.append(plt.text(sangle1, tangle,'({:.1f},{:.1f})'.format(sangle1,tangle), size=12, horizontalalignment='center', color='r', weight='bold'))
            fig1_text.append(plt.text(sangle2, -tangle,'({:.1f},{:.1f})'.format(sangle2,-tangle), size=12, horizontalalignment='center', color='r', weight='bold'))

            plt.xlabel('σ')
            plt.ylabel('τ')
            #plt.title("Mohr's Circle")
            plt.grid(color='b', linestyle=':', linewidth=0.5)
        with self.plot_output:
            adjust_text(fig1_text, expand_points=(1.5, 1.5),ax=ax)
            plt.show()

        fig2 = plt.figure(figsize=[6, 6])
        sqr = patches.Rectangle((-1, -1), 2.,2., linewidth=2, edgecolor='k', fill=False)
        t_tp = patches.FancyArrowPatch((0,1.2),(0,1.7),connectionstyle='arc3, rad=0.0', **kw)
        t_bp = patches.FancyArrowPatch((0,-1.2),(0,-1.7),connectionstyle='arc3, rad=0.0', **kw)
        t_lp = patches.FancyArrowPatch((-1.2,0),(-1.7,0),connectionstyle='arc3, rad=0.0', **kw)
        t_rp = patches.FancyArrowPatch((1.2,0),(1.7,0),connectionstyle='arc3, rad=0.0', **kw)

        t_tn = patches.FancyArrowPatch((0,1.7),(0,1.2),connectionstyle='arc3, rad=0.0', **kw)
        t_bn = patches.FancyArrowPatch((0,-1.7),(0,-1.2),connectionstyle='arc3, rad=0.0', **kw)
        t_ln = patches.FancyArrowPatch((-1.7,0),(-1.2,0),connectionstyle='arc3, rad=0.0', **kw)
        t_rn = patches.FancyArrowPatch((1.7,0),(1.2,0),connectionstyle='arc3, rad=0.0', **kw)

        s_tp = patches.FancyArrowPatch((-0.8,1.1),(0.8,1.1),connectionstyle='arc3, rad=0.0', **kw)
        s_bp = patches.FancyArrowPatch((0.8,-1.1),(-0.8,-1.1),connectionstyle='arc3, rad=0.0', **kw)
        s_lp = patches.FancyArrowPatch((-1.1,0.8),(-1.1,-0.8),connectionstyle='arc3, rad=0.0', **kw)
        s_rp = patches.FancyArrowPatch((1.1,-0.8),(1.1,0.8),connectionstyle='arc3, rad=0.0', **kw)

        s_tn = patches.FancyArrowPatch((0.8,1.1),(-0.8,1.1),connectionstyle='arc3, rad=0.0', **kw)
        s_bn = patches.FancyArrowPatch((-0.8,-1.1),(0.8,-1.1),connectionstyle='arc3, rad=0.0', **kw)
        s_ln = patches.FancyArrowPatch((-1.1,-0.8),(-1.1,0.8),connectionstyle='arc3, rad=0.0', **kw)
        s_rn = patches.FancyArrowPatch((1.1,0.8),(1.1,-0.8),connectionstyle='arc3, rad=0.0', **kw)

        style1 = "Fancy, tail_width=0.5, head_width=3, head_length=5"
        kw1 = dict(arrowstyle=style1, color="blue")
        kw2 = dict(arrowstyle=style1, color="red")
        x_ax = patches.FancyArrowPatch((1.1,-1),(2,-1),connectionstyle='arc3, rad=0.0', **kw1)
        y_ax = patches.FancyArrowPatch((-1,1.1),(-1,2),connectionstyle='arc3, rad=0.0', **kw1)
        xp_ax = patches.FancyArrowPatch((-1,-1),(0.5,0.5),connectionstyle='arc3, rad=0.0',linestyle=':', **kw2)
        coord_ang = patches.FancyArrowPatch((-0.3,-1),(-0.5,-0.5),connectionstyle='arc3, rad=0.5', **kw0)

        plt.clf()
        plt.axis('image')
        plt.axis('off')
        ax2 = plt.gca() 
        ax2.set_xlim(-3,3)
        ax2.set_ylim(-3,3)
        ax2.set_aspect(1)
        ax2.add_artist(sqr) 
        if sy < 0.0:
            ax2.add_artist(t_tn) 
            ax2.add_artist(t_bn)
        else:
            ax2.add_artist(t_tp) 
            ax2.add_artist(t_bp)
        if sx < 0.0:
            ax2.add_artist(t_ln) 
            ax2.add_artist(t_rn) 
        else:
            ax2.add_artist(t_lp) 
            ax2.add_artist(t_rp) 
        if txy < 0.0: # at rhs surface of cube in x normal surface
            ax2.add_artist(s_tp) 
            ax2.add_artist(s_bp) 
            ax2.add_artist(s_lp) 
            ax2.add_artist(s_rp) 
        else:
            ax2.add_artist(s_tn) 
            ax2.add_artist(s_bn) 
            ax2.add_artist(s_ln) 
            ax2.add_artist(s_rn) 
        ax2.add_artist(x_ax) 
        ax2.add_artist(y_ax) 
        ax2.add_artist(xp_ax) 
        ax2.add_artist(coord_ang)
        plt.text(2.1,-1,'X',size=12)
        plt.text(-1, 2.1,'Y',size=12)
        plt.text(0.2, 0.5,'X\'',color='red',size=12)
        plt.text(-0.2, -0.8,'+θ',color='red',size=12)

        plt.text(1.5, 0.2,'{:.1f}'.format(abs(sx)),size=12)
        plt.text(-2.2, 0.2,'{:.1f}'.format(abs(sx)),size=12)
        plt.text(0.0, 1.8,'{:.1f}'.format(abs(sy)),size=12)
        plt.text(0.0, -2.0,'{:.1f}'.format(abs(sy)),size=12)
        plt.text(1.1, 1.1,'{:.1f}'.format(abs(txy)),size=12)
        plt.text(-1.5, -1.4,'{:.1f}'.format(abs(txy)),size=12)
        plt.text(-0.7, -2.5,'INITIAL'.format(abs(tangle)),size=15, weight='bold')

        with self.plot_output2:
            plt.show()

        fig3 = plt.figure(figsize=[6, 6])

        xy = np.array([rotateAxisZ(angl,[-1,-1]), rotateAxisZ(angl,[1,-1]), rotateAxisZ(angl,[1,1]), rotateAxisZ(angl,[-1,1])])
        sqr = patches.Polygon(xy, edgecolor='k', facecolor='white', linewidth=2.0, closed=True)

        t_tp = patches.FancyArrowPatch(rotateAxisZ(angl,(0,1.2)),rotateAxisZ(angl,(0,1.7)),connectionstyle='arc3, rad=0.0', **kw)
        t_bp = patches.FancyArrowPatch(rotateAxisZ(angl,(0,-1.2)),rotateAxisZ(angl,(0,-1.7)),connectionstyle='arc3, rad=0.0', **kw)
        t_lp = patches.FancyArrowPatch(rotateAxisZ(angl,(-1.2,0)),rotateAxisZ(angl,(-1.7,0)),connectionstyle='arc3, rad=0.0', **kw)
        t_rp = patches.FancyArrowPatch(rotateAxisZ(angl,(1.2,0)),rotateAxisZ(angl,(1.7,0)),connectionstyle='arc3, rad=0.0', **kw)

        t_tn = patches.FancyArrowPatch(rotateAxisZ(angl,(0,1.7)),rotateAxisZ(angl,(0,1.2)),connectionstyle='arc3, rad=0.0', **kw)
        t_bn = patches.FancyArrowPatch(rotateAxisZ(angl,(0,-1.7)),rotateAxisZ(angl,(0,-1.2)),connectionstyle='arc3, rad=0.0', **kw)
        t_ln = patches.FancyArrowPatch(rotateAxisZ(angl,(-1.7,0)),rotateAxisZ(angl,(-1.2,0)),connectionstyle='arc3, rad=0.0', **kw)
        t_rn = patches.FancyArrowPatch(rotateAxisZ(angl,(1.7,0)),rotateAxisZ(angl,(1.2,0)),connectionstyle='arc3, rad=0.0', **kw)

        s_tp = patches.FancyArrowPatch(rotateAxisZ(angl,(-0.8,1.1)),rotateAxisZ(angl,(0.8,1.1)),connectionstyle='arc3, rad=0.0', **kw)
        s_bp = patches.FancyArrowPatch(rotateAxisZ(angl,(0.8,-1.1)),rotateAxisZ(angl,(-0.8,-1.1)),connectionstyle='arc3, rad=0.0', **kw)
        s_lp = patches.FancyArrowPatch(rotateAxisZ(angl,(-1.1,0.8)),rotateAxisZ(angl,(-1.1,-0.8)),connectionstyle='arc3, rad=0.0', **kw)
        s_rp = patches.FancyArrowPatch(rotateAxisZ(angl,(1.1,-0.8)),rotateAxisZ(angl,(1.1,0.8)),connectionstyle='arc3, rad=0.0', **kw)

        s_tn = patches.FancyArrowPatch(rotateAxisZ(angl,(0.8,1.1)),rotateAxisZ(angl,(-0.8,1.1)),connectionstyle='arc3, rad=0.0', **kw)
        s_bn = patches.FancyArrowPatch(rotateAxisZ(angl,(-0.8,-1.1)),rotateAxisZ(angl,(0.8,-1.1)),connectionstyle='arc3, rad=0.0', **kw)
        s_ln = patches.FancyArrowPatch(rotateAxisZ(angl,(-1.1,-0.8)),rotateAxisZ(angl,(-1.1,0.8)),connectionstyle='arc3, rad=0.0', **kw)
        s_rn = patches.FancyArrowPatch(rotateAxisZ(angl,(1.1,0.8)),rotateAxisZ(angl,(1.1,-0.8)),connectionstyle='arc3, rad=0.0', **kw)

        style1 = "Fancy, tail_width=0.5, head_width=3, head_length=5"
        kw1 = dict(arrowstyle=style1, color="blue")
        kw2 = dict(arrowstyle=style1, color="red")
        x_ax = patches.FancyArrowPatch(rotateAxisZ(angl,(1.1,-1)),rotateAxisZ(angl,(2,-1)),connectionstyle='arc3, rad=0.0', **kw2)
        y_ax = patches.FancyArrowPatch(rotateAxisZ(angl,(-1,1.1)),rotateAxisZ(angl,(-1,2)),connectionstyle='arc3, rad=0.0', **kw2)

        plt.clf()
        plt.axis('image')
        plt.axis('off')
        ax2 = plt.gca() 
        ax2.set_xlim(-3,3)
        ax2.set_ylim(-3,3)
        ax2.set_aspect(1)
        ax2.add_artist(sqr) 
        if sangle2 < 0.0:
            ax2.add_artist(t_tn) 
            ax2.add_artist(t_bn)
        else:
            ax2.add_artist(t_tp) 
            ax2.add_artist(t_bp)
        if sangle1 < 0.0:
            ax2.add_artist(t_ln) 
            ax2.add_artist(t_rn) 
        else:
            ax2.add_artist(t_lp) 
            ax2.add_artist(t_rp) 
        if tangle < 0.0: # at rhs surface of cube in x normal surface
            ax2.add_artist(s_tp) 
            ax2.add_artist(s_bp) 
            ax2.add_artist(s_lp) 
            ax2.add_artist(s_rp) 
        else:
            ax2.add_artist(s_tn) 
            ax2.add_artist(s_bn) 
            ax2.add_artist(s_ln) 
            ax2.add_artist(s_rn) 
        ax2.add_artist(x_ax) 
        ax2.add_artist(y_ax) 

        coord = rotateAxisZ(angl,(2.1, -1))
        plt.text(coord[0], coord[1],'X\'',color='red',size=12)
        coord = rotateAxisZ(angl,(-1, 2.1))
        plt.text(coord[0], coord[1],'Y\'',color='red',size=12)
        coord = rotateAxisZ(angl,(1.5, 0.2))
        plt.text(coord[0], coord[1],'{:.1f}'.format(abs(sangle1)),size=12)
        coord = rotateAxisZ(angl,(-2.2, 0.2))
        plt.text(coord[0], coord[1],'{:.1f}'.format(abs(sangle1)),size=12)
        coord = rotateAxisZ(angl,(0.0, 1.8))
        plt.text(coord[0], coord[1],'{:.1f}'.format(abs(sangle2)),size=12)
        coord = rotateAxisZ(angl,(0.0, -2.0))
        plt.text(coord[0], coord[1],'{:.1f}'.format(abs(sangle2)),size=12)
        coord = rotateAxisZ(angl,(1.1, 1.1))
        plt.text(coord[0], coord[1],'{:.1f}'.format(abs(tangle)),size=12)
        coord = rotateAxisZ(angl,(-1.5, -1.4))
        plt.text(coord[0], coord[1],'{:.1f}'.format(abs(tangle)),size=12)

        plt.text(-1.2, -2.5,'TRANSFORMED'.format(abs(tangle)),size=15, weight='bold')

        with self.plot_output3:
            plt.show()

        # calculate design results
        self.imaxS1, self.maxS1 = get_max_min_index(self.s1, 'Max')
        self.iminS1, self.minS1 = get_max_min_index(self.s1, 'Min')
        self.imaxS2, self.maxS2 = get_max_min_index(self.s2, 'Max')
        self.iminS2, self.minS2 = get_max_min_index(self.s2, 'Min')
        self.imaxSxp, self.maxSxp = get_max_min_index(self.sxp, 'Max')
        self.iminSxp, self.minSxp = get_max_min_index(self.sxp, 'Min')
        self.imaxSyp, self.maxSyp = get_max_min_index(self.syp, 'Max')
        self.iminSyp, self.minSyp = get_max_min_index(self.syp, 'Min')
        self.imaxTxyp, self.maxTxyp = get_max_min_index(self.txyp, 'Max')
        self.iminTxyp, self.minTxyp = get_max_min_index(self.txyp, 'Min')   
        
    def ui_mohrcircle2d(self, asx, asy, atxy, unit, angle):
        
        idx = 0
        dim_index = 0
        
        self.sx = asx
        self.sy = asy
        self.txy = atxy
        self.angle = angle
        self.arr_shape = asx.shape

        var_list = [item for item in dir() if not (item.startswith("__") or item.startswith("_")) ]
       
        if len(asx.shape)==1:
            dim_index = 1
            i_idx_list = list(range(len(asx)))
            j_idx_list = [0]
            k_idx_list = [0]
        elif len(asx.shape)==2:   
            dim_index = 2
            i_idx_list = list(range(len(asx[:,0])))
            j_idx_list = list(range(len(asx[0,:])))
            k_idx_list = [0]
        elif len(asx.shape)==3:     
            dim_index = 3
            i_idx_list = list(range(len(asx[:,0,0])))
            j_idx_list = list(range(len(asx[0,:,0])))
            k_idx_list = list(range(len(asx[0,0,:])))

        v_sx = asx.flatten()
        v_sy = asy.flatten()
        v_txy = atxy.flatten()
        v_angle = angle.flatten()

        in_out = widgets.Output()
        with in_out:
            display(display(IFrame('https://pyenote.web.app/Mohr_Circle_2D.html', 1000,600)))
        manual_forum = widgets.Output()
        with manual_forum:
            button1 = widgets.Button(description="Open Manual")
            display(button1)
            button2 = widgets.Button(description="Open Forum")
            display(button2)
           # display(Javascript(f'window.open("https://sites.google.com/view/pyenote-home/general-stress-analysis/mohrs-circle-2d?authuser=0");'))
           # display(IFrame("https://sites.google.com/view/pyenote-home/general-stress-analysis/mohrs-circle-2d?authuser=0", 1200,600))


        items_layout = widgets.Layout(display='flex', flex_flow='row wrap', align_items='center', align_content='stretch', width='90%', margin='10px 10px 10px 10px',) 
        index_layout = widgets.Layout(display='flex', flex_flow='center', align_items='center', align_content='stretch', width='auto', height='100%', margin='10px 5x 10px 5x ') 
        input_layout = widgets.Layout(flex='1 1 auto', display='flex', align_items='flex-start', flex_flow='column', max_width='95%', width='auto', margin='30px 10px 10px 10px')  
        tab_layout = widgets.Layout(display='flex', margin='10px 0 20px 0',width='95%', height='100%', overflow_y='auto')

        box_layout = widgets.Layout(display='flex', flex_flow='row wrap', align_items='center', 
                                    align_content='flex-start', width='100%', height='100%', justify_content='center',
                                    border='solid 1px', padding='0 10px 0 10px')

        def create_FloatText(value, amax, amin, description):
            return widgets.FloatText(value=value,  step=1, description=description,disabled=False, layout=items_layout)

        find = widgets.Dropdown(options=['', \
                                         'maxS1: Max. of Max. Principal Stress (σ1)',\
                                         'minS1: Min. of Max. Principal Stress (σ1)',\
                                         'maxS2: Max. of Min. Principal Stress (σ2)',\
                                         'minS2: Min. of Min. Principal Stress (σ2)',\
                                         'maxSxp: Max. of Transformed Stress (σx\')',\
                                         'minSxp: Min. of Transformed Stress (σx\')',\
                                         'maxSyp: Max. of Transformed Stress (σy\')',\
                                         'minSyp: Min. of Transformed Stress (σy\')',\
                                         'maxTxyp: Max. of Transformed Stress (τx\'y\')',\
                                         'minTxyp: Min. of Transformed Stress (τx\'y\')'],\
                                value='',description='Find: ',disabled=False, layout=items_layout)
        index_i = widgets.Dropdown(value=0, options=i_idx_list, description='I:',disabled=False, layout=index_layout)
        index_j = widgets.Dropdown(value=0, options=j_idx_list, description='J:',disabled=False, layout=index_layout)
        index_k = widgets.Dropdown(value=0, options=k_idx_list, description='K:',disabled=False, layout=index_layout)
        index = widgets.HBox([index_i, index_j, index_k],layout=items_layout)
        elem_disp = widgets.HBox([self.plot_output2, self.plot_output3], layout=index_layout)

        sx = create_FloatText(v_sx[idx], 1000., -1000., 'σx: ')
        sy = create_FloatText(v_sy[idx], 1000., -1000., 'σy: ')
        txy = create_FloatText(v_txy[idx], 1000., -1000., 'τxy: ')   
        ang = create_FloatText(v_angle[idx], 179.9, -179.9, 'Angle(Deg): ') 

        inputs = widgets.VBox([sx, sy, txy, ang, find, index, elem_disp], layout=input_layout)
        
        idx = index_array2vector((index_i.value,index_j.value, index_k.value),asx.shape)

        self.mohrcircle2d(v_sx,v_sy,v_txy,find.value,v_angle,idx,False)

        def sx_eventhandler(change):
            find.unobserve(find_eventhandler, names='value')
            find.value = ''
            find.observe(find_eventhandler, names='value')

            self.mohrcircle2d(change.new,sy.value,txy.value,find.value,ang.value,idx,True)

        def sy_eventhandler(change):
            find.unobserve(find_eventhandler, names='value')
            find.value = ''
            find.observe(find_eventhandler, names='value')

            self.mohrcircle2d(sx.value,change.new,txy.value,find.value,ang.value,idx,True)

        def txy_eventhandler(change):
            find.unobserve(find_eventhandler, names='value')
            find.value = ''
            find.observe(find_eventhandler, names='value')

            self.mohrcircle2d(sx.value,sy.value,change.new,find.value,ang.value,idx,True)

        def find_eventhandler(change):
            sx.unobserve(sx_eventhandler, names='value')
            sy.unobserve(sy_eventhandler, names='value')
            txy.unobserve(txy_eventhandler, names='value')
            ang.unobserve(ang_eventhandler, names='value')
            index_i.unobserve(index_eventhandler, names='value')
            index_j.unobserve(index_eventhandler, names='value')
            index_k.unobserve(index_eventhandler, names='value')

            keyword = change.new.split(':')[0]
            if keyword == 'maxS1':
                id_key = self.imaxS1
            elif keyword == 'minS1':
                id_key = self.iminS1
            elif keyword == 'maxS2':
                id_key = self.imaxS2
            elif keyword == 'minS2':
                id_key = self.iminS2
            elif keyword == 'maxSxp':
                id_key = self.imaxSxp
            elif keyword == 'minSxp':
                id_key = self.iminSxp
            elif keyword == 'maxSyp':
                id_key = self.imaxSyp
            elif keyword == 'minSyp':
                id_key = self.iminSyp
            elif keyword == 'maxTxyp':
                id_key = self.imaxTxyp
            elif keyword == 'minTxyp':
                id_key = self.iminTxyp  
            
            if keyword =='':
                sx.observe(sx_eventhandler, names='value')
                sy.observe(sy_eventhandler, names='value')
                txy.observe(txy_eventhandler, names='value')
                ang.observe(ang_eventhandler, names='value')
                index_i.observe(index_eventhandler, names='value')
                index_j.observe(index_eventhandler, names='value')
                index_k.observe(index_eventhandler, names='value')
            else:
                idx = index_array2vector(id_key,asx.shape)
                sx.value = v_sx[idx]
                sy.value = v_sy[idx]
                txy.value = v_txy[idx]
                ang.value = v_angle[idx]
                if len(asx.shape)==3:
                    index_i.value = id_key[0]
                    index_j.value = id_key[1]
                    index_k.value = id_key[2]
                elif len(asx.shape)==2:
                    index_i.value = id_key[0]
                    index_j.value = id_key[1]
                    index_k.value = 0
                elif len(asx.shape)==1:
                    index_i.value = id_key[0]
                    index_j.value = 0
                    index_k.value = 0
                    
                sx.observe(sx_eventhandler, names='value')
                sy.observe(sy_eventhandler, names='value')
                txy.observe(txy_eventhandler, names='value')
                ang.observe(ang_eventhandler, names='value')
                index_i.observe(index_eventhandler, names='value')
                index_j.observe(index_eventhandler, names='value')
                index_k.observe(index_eventhandler, names='value')
                
                idx = index_array2vector((index_i.value,index_j.value, index_k.value),asx.shape)

                self.id_i = index_i.value
                self.id_j = index_j.value
                self.id_k = index_k.value

                self.mohrcircle2d(v_sx,v_sy,v_txy,find.value,v_angle,idx,False)


        def ang_eventhandler(change):
            find.unobserve(find_eventhandler, names='value')
            find.value = ''
            find.observe(find_eventhandler, names='value')

            self.mohrcircle2d(sx.value,sy.value,txy.value,find.value,change.new,idx,True)

        def index_eventhandler(change):
            sx.unobserve(sx_eventhandler, names='value')
            sy.unobserve(sy_eventhandler, names='value')
            txy.unobserve(txy_eventhandler, names='value')
            find.unobserve(find_eventhandler, names='value')
            ang.unobserve(ang_eventhandler, names='value')

            if change.owner.description == 'I:':
                idx = index_array2vector((change.new,index_j.value, index_k.value),asx.shape)
            elif change.owner.description == 'J:':
                idx = index_array2vector((index_i.value,change.new, index_k.value),asx.shape)
            elif  change.owner.description == 'K:':
                idx = index_array2vector((index_i.value,index_j.value, change.new),asx.shape)
            
            sx.value = v_sx[idx]
            sy.value = v_sy[idx]
            txy.value = v_txy[idx]
            ang.value = v_angle[idx]
            find.value = ''

            sx.observe(sx_eventhandler, names='value')
            sy.observe(sy_eventhandler, names='value')
            txy.observe(txy_eventhandler, names='value')
            find.observe(find_eventhandler, names='value')
            ang.observe(ang_eventhandler, names='value')

            self.mohrcircle2d(v_sx,v_sy,v_txy,find.value,v_angle,idx,False)

        sx.observe(sx_eventhandler, names='value')
        sy.observe(sy_eventhandler, names='value')
        txy.observe(txy_eventhandler, names='value')
        find.observe(find_eventhandler, names='value')
        ang.observe(ang_eventhandler, names='value')
        index_i.observe(index_eventhandler, names='value')
        index_j.observe(index_eventhandler, names='value')
        index_k.observe(index_eventhandler, names='value')

        def on_clicked1(a):
            # generate an URL
            url = 'https://sites.google.com/view/pyenote-home/general-stress-analysis/mohrs-circle-2d?authuser=0'
            webbrowser.open(url)
            #display(Javascript('window.open("{url}");'.format(url=url)))
        def on_clicked2(a):
            # generate an URL
            url = 'https://www.reddit.com/r/pyairframe/comments/vj6dbp/welcome_to_pyairframe_user_community/'
            webbrowser.open(url)
            #display(Javascript('window.open("{url}");'.format(url=url)))

        button1.on_click(on_clicked1)
        button2.on_click(on_clicked2)

        dashboard = widgets.Box([inputs, self.plot_output], layout=box_layout)

        tab = widgets.Tab([dashboard, in_out, manual_forum],layout=tab_layout)
        tab.set_title(0, 'Mohr Circle 2D')
        tab.set_title(1, 'Manual/Tutorial')        
        tab.set_title(2, 'Forum')              

        display(tab)

    def display_equation(self):
        for i in self.print_equation:
            display(Math(i))
