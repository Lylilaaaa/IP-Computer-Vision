import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ginput
from scipy.optimize import fsolve

def getVanishingPoint(im):
    lines = np.zeros((0, 3))
    line_length = []
    centers = np.zeros((0, 3))

    plt.imshow(im)

    while True:
        print(' ')
        print('Click first point or click the same point twice to stop')
        
        x1,y1 = ginput(1)[0]
        
##        if b=='q':      
##            break
        
        print('Click second point')
        x2,y2 = ginput(1)[0]
        #draw the blue line
        plt.plot([x1, x2], [y1, y2], 'b')

        length = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        if length < 0.0001:
            break
        
        lines = np.vstack([lines, np.cross(np.array([x1, y1, 1]).reshape(1, 3),
                                   np.array([x2, y2, 1]).reshape(1, 3))])
        
        line_length.append(length)
        centers = np.vstack([centers, np.array([x1+x2, y1+y2, 2]).reshape(1, 3)/2])

    print('find VP')
    # insert code here to compute vp (3-d vector in homogeneous coordinates)

    #find vps
    vpArray = []
    for i in range(0,lines.shape[0],2):
        line1 = lines[i]
        line2 = lines[i+1]
        vp = np.cross(line1,line2) #(1,3)
        print('The ',int(i/2),'th vp(u,v) is: ','[',round(vp[0]/vp[2],3),',',round(vp[1]/vp[2],3),']')
        vpArray.append(vp)

    def show2LinesVP(vp,_lines):
        bx1 = min(1, vp[0] / vp[2]) - 10
        bx2 = max(im.shape[1], vp[0] / vp[2]) + 10
        #for calculating the height in (d), we make constraints for the size of the images
        by1 = max(min(1, vp[1] / vp[2]) - 10,-50)
        by2 = min(max(im.shape[0], vp[1] / vp[2]) + 10,im.shape[0]+50)
        for k  in range(_lines.shape[0]): #check each lines
            if np.abs(_lines[k, 0]) < np.abs(_lines[k, 1]):
                pt1 = np.cross(np.array([1, 0, -bx1]).reshape(1, 3), _lines[k]).reshape(3,1)
                pt2 = np.cross(np.array([1, 0, -bx2]).reshape(1, 3), _lines[k]).reshape(3,1)
            else:
                pt1 = np.cross(np.array([0, 1, -by1]).reshape(1, 3), _lines[k]).reshape(3,1)
                pt2 = np.cross(np.array([0, 1, -by2]).reshape(1, 3), _lines[k]).reshape(3,1)
            pt1 = pt1 / pt1[2]
            pt2 = pt2 / pt2[2]
            
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g', 'Linewidth', 1)
        pass
    
    for dim in range(3):
         show2LinesVP(vpArray[dim],lines[dim*2:dim*2+2,:])
    
    #find the vanishing line
    def drawHorizon(_vpArray):
        vpx = _vpArray[0]
        vpy = _vpArray[1]
        plt.plot([vpx[0]/vpx[2], vpy[0]/vpy[2]],[vpx[1]/vpx[2], vpy[1]/vpy[2]], 'b', 'Linewidth', 1)

    def lineHorizon(_vpArray):
        vpx = _vpArray[0]
        vpy = _vpArray[1]
        para = np.cross(vpx, vpy)
        a = para[0]
        b = para[1]
        c = para[2]
        divider = np.sqrt(a**2+b**2)
        a = a/divider
        b = b/divider
        print('The ground horizon line is in the form: ',a,'*u + ',b,'*v + ',c,'= 0')
    
    drawHorizon(vpArray)
    lineHorizon(vpArray)

    #find the parameter of the camera
    _vp1 = vpArray[0]
    _vp2 = vpArray[1]
    _vp3 = vpArray[2]
    vp1 = [_vp1[0]/_vp1[2],_vp1[1]/_vp1[2]]
    vp2 = [_vp2[0]/_vp2[2],_vp2[1]/_vp2[2]]
    vp3 = [_vp3[0]/_vp3[2],_vp3[1]/_vp3[2]]

    u = (vp1[1]*(vp2[1]**2)-(vp1[1]**2)*vp2[1]-vp1[1]*(vp3[1]**2)+(vp1[1]**2)*vp3[1]+vp2[1]*(vp3[1])**2-(vp2[1]**2)*vp3[1]-vp1[0]*vp2[0]*vp1[1]+\
        vp1[0]*vp2[0]*vp2[1]+vp1[0]*vp3[0]*vp1[1]-vp1[0]*vp3[0]*vp3[1]-vp2[0]*vp3[0]*vp2[1]+vp2[0]*vp3[0]*vp3[1])/\
        (vp1[0]*vp2[1]-vp2[0]*vp1[1]-vp1[0]*vp3[1]+vp3[0]*vp1[1]+vp2[0]*vp3[1]-vp3[0]*vp2[1])
    v = (vp2[0]*(vp1[0]-vp3[0])-u*(vp1[0]-vp3[0])+vp2[1]*(vp1[1]-vp3[1]))/(vp1[1]-vp3[1])
    f = np.sqrt(np.abs(-(vp1[0]-u)*(vp2[0]-u)-(vp1[1]-v)*(vp2[1]-v)))
    print('[u0,v0] = ','[',u,',',v,']')
    print('f = ',f)

    #find the rotation matrix
    R1 = [(_vp1[0]-_vp1[2]*u)/f,(_vp2[0]-_vp2[2]*u)/f,(_vp3[0]-_vp3[2]*u)/f]
    R2 = [(_vp1[1]-_vp1[2]*u)/f,(_vp2[1]-_vp2[2]*u)/f,(_vp3[1]-_vp3[2]*u)/f]
    R3 = [_vp1[2],_vp2[2],_vp3[2]]
    R = np.array([R1,R2,R3])
    print('The rotation matrix R is: \n',R)
    return vp1,vp2,vp3

#find the height of the tractor, building and the camera
def findheight(vp1,vp2,vp3):
    hori_line = np.cross([vp1[0],vp1[1],1],[vp2[0],vp2[1],1])

    print('Please click the bottom of the sign')
    xs_b,ys_b = ginput(1)[0]
    print('Please click the top of the sign')
    xs_t,ys_t = ginput(1)[0]
    plt.plot([xs_b, xs_t], [ys_b, ys_t], 'b', 'Linewidth', 1)

    #find the height of the tractor
    print('Please click the bottom of the tractor')
    xt_b,yt_b = ginput(1)[0]
    print('Please click the top of the tractor')
    xt_t,yt_t = ginput(1)[0]
    plt.plot([xt_b, xt_t], [yt_b, yt_t], 'b','Linewidth', 1)

    bottom_line = np.cross([xs_b,ys_b,1],[xt_b,yt_b,1])
    hori_pt = np.cross(bottom_line,hori_line)
    hori_pt_n = [hori_pt[0]/hori_pt[2],hori_pt[1]/hori_pt[2]]
    #plot the bottom line
    plt.plot([hori_pt_n[0], xt_b], [hori_pt_n[1], yt_b], 'r', 'Linewidth', 1)
    top_line = np.cross([hori_pt_n[0],hori_pt_n[1],1],[xt_t,yt_t,1])
    #plot the top line
    plt.plot([hori_pt_n[0], xt_t], [hori_pt_n[1], yt_t], 'r', 'Linewidth', 1)
    sign_line = np.cross([xs_t,ys_t,1],[xs_b,ys_b,1])
    sign_cross = np.cross(sign_line,top_line)
    sign_cross_n = [sign_cross[0]/sign_cross[2],sign_cross[1]/sign_cross[2]]
    h_prime = np.sqrt(np.abs((sign_cross_n[1]-ys_b)**2 + (sign_cross_n[0]-xs_b)**2))
    h_s = np.sqrt(np.abs((ys_t-ys_b)**2+(xs_t-xs_b)**2))
    h_s_inf = np.sqrt(np.abs((vp3[1]-ys_t)**2+(vp3[0]-xs_t)**2))
    h_prime_inf = np.sqrt(np.abs((vp3[1]-sign_cross_n[1])**2+(vp3[0]-sign_cross_n[0])**2))

    height_t = 1.65*h_prime*(h_s_inf+h_s)/(h_s*h_prime_inf)

    #find the height of the building
    print('Please click the bottom of the building')
    xb_b,yb_b = ginput(1)[0]
    print('Please click the top of the building')
    xb_t,yb_t = ginput(1)[0]
    plt.plot([xb_b, xb_t], [yb_b, yb_t], 'b','Linewidth', 1)

    bottom_line_b = np.cross([xs_b,ys_b,1],[xb_b,yb_b,1])
    hori_pt_b = np.cross(bottom_line_b,hori_line)
    hori_pt_n_b = [hori_pt_b[0]/hori_pt_b[2],hori_pt_b[1]/hori_pt_b[2]]
    #plot the bottom line
    plt.plot([hori_pt_n_b[0], xb_b], [hori_pt_n_b[1], yb_b], 'r', 'Linewidth', 1)
    top_line_b = np.cross([hori_pt_n_b[0],hori_pt_n_b[1],1],[xb_t,yb_t,1])
    #plot the top line
    plt.plot([hori_pt_n_b[0], xb_t], [hori_pt_n_b[1], yb_t], 'r', 'Linewidth', 1)
    sign_cross_b = np.cross(sign_line,top_line_b)
    sign_cross_n_b = [sign_cross_b[0]/sign_cross_b[2],sign_cross_b[1]/sign_cross_b[2]]
    h_prime_b = np.sqrt(np.abs((sign_cross_n_b[1]-ys_b)**2 + (sign_cross_n_b[0]-xs_b)**2))
    h_prime_inf_b = np.sqrt(np.abs((vp3[1]-sign_cross_n_b[1])**2+(vp3[0]-sign_cross_n_b[0])**2))

    height_b = 1.65*h_prime_b*(h_s_inf+h_s)/(h_s*h_prime_inf_b)
    print('The height of the tractor is: ',height_t)
    print('The height of the building is: ',height_b)

    #find the height of the camera
    sign_cross_c = np.cross(sign_line,hori_line)
    sign_cross_n_c = [sign_cross_c[0]/sign_cross_c[2],sign_cross_c[1]/sign_cross_c[2]]
    h_prime_c = np.sqrt(np.abs((sign_cross_n_c[1]-ys_b)**2 + (sign_cross_n_c[0]-xs_b)**2))
    h_prime_inf_c = np.sqrt(np.abs((vp3[1]-sign_cross_n_c[1])**2+(vp3[0]-sign_cross_n_c[0])**2))

    height_c = 1.65*h_prime_c*(h_s_inf+h_s)/(h_s*h_prime_inf_c)
    print('The height of the camera is: ',height_c)

if __name__ == "__main__":
    im = plt.imread('./kyoto_street.JPG')
    getVanishingPoint(im)
    plt.show()
    
    _im = plt.imread('./CIMG6476.JPG')
    v1,v2,v3 = getVanishingPoint(_im)
    findheight(v1,v2,v3)
    plt.show()