properties([
    pipelineTriggers([githubPush()])
//     pipelineTriggers([githubPullRequests()])
])

// // Not figured out yet
// properties([
//     pipelineTriggers([
//         [
//             $class: 'GitHubPRTrigger',
//             events: [
//                 [$class: 'GitHubPROpenEvent']
//             ]
//         ]
//     ])
// ])

pipeline {
    agent any
    tools {
        git 'Default'  // Uses the Git tool configured in Jenkins Global Tools
    }
    stages {
        stage('Checkout PR') {
            steps {
                sh 'echo "Start PR checkout"'
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: 'refs/heads/develop']], // ${sha1}
                    extensions: [
                        [$class: 'CloneOption', depth: 1, shallow: true]
                    ],
                    userRemoteConfigs: [[
                        url: 'https://github.com/Camille0512/NLA.git',
                        credentialsId: 'jenkins_nla',
                        refspec: '+refs/pull/*:refs/remotes/origin/pr/*'
                    ]]
                ])
                sh 'echo "Finish PR checkout"'
            }
        }

        stage('Build & Test') {
            steps {
                sh '''
                    python3 -m pip install --upgrade pip
                    pip3 install pytest
                    python3 -m pytest --junitxml=./JenkinsLogs/surefire-reports/test-results.xml
                '''
                junit '**/JenkinsLogs/surefire-reports/*.xml'
                sh 'echo "Finish Build & Test"'
            }
        }
    }
}

post {
    always {
        cleanWs()  // Clean workspace after build
    }
}